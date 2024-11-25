import pickle
import jax
import jax.numpy as jnp
from pathlib import Path

import train
from typing import Optional, Sequence, Callable
from train import SelfplayOutput, Embeddings, Checksums
import dnd5e
from flax import nnx
from train import MLP
from tree_serialization import flatten_pytree_batched
import optax
import mctx
import chex
import pygraphviz


def convert_tree_to_graph(
        tree: mctx.Tree,
        action_labels: Optional[Callable] = None,
        batch_index: int = 0
) -> pygraphviz.AGraph:
    """Converts a search tree into a Graphviz graph.

    Args:
      tree: A `Tree` containing a batch of search data.
      action_labels: Optional labels for edges, defaults to the action index.
      batch_index: Index of the batch element to plot.

    Returns:
      A Graphviz graph representation of `tree`.
    """
    chex.assert_rank(tree.node_values, 2)
    batch_size = tree.node_values.shape[0]

    def node_to_str(node_i, reward=0, discount=1):
        terminated = tree.embeddings.terminated[batch_index, node_i].item()
        color = "red" if terminated else "blue"
        return (f"{node_i}\n"
                f"Reward: {reward:.2f}\n"
                f"Discount: {discount:.2f}\n"
                f"Value: {tree.node_values[batch_index, node_i]:.2f}\n"
                f"Raw Value: {tree.raw_values[batch_index, node_i]:.2f}\n"
                f"Visits: {tree.node_visits[batch_index, node_i]}\n"
                f"Terminal: {terminated}\n"
                ), color

    def edge_to_str(node_i, a_i):
        node_index = jnp.full([batch_size], node_i)
        probs = jax.nn.softmax(tree.children_prior_logits[batch_index, node_i])
        current_player = tree.embeddings.current_player[batch_index, node_i]
        pos = tree.embeddings.pos[batch_index, node_i]
        action_tuple = dnd5e.decode_action(a_i, current_player, pos)
        action_str = dnd5e.repr_action(action_tuple)
        return (f"{a_i} {action_str}\n"
                f"Q: {tree.qvalues(node_index)[batch_index, a_i]:.2f}\n"  # pytype: disable=unsupported-operands  # always-use-return-annotations
                f"p: {probs[a_i]:.2f}\n")

    graph = pygraphviz.AGraph(directed=True)

    # Add root
    graph.add_node(0, label=node_to_str(node_i=0)[0], color="green")
    # Add all other nodes and connect them up.
    for node_i in range(tree.num_simulations):
        for a_i in range(tree.num_actions):
            # Index of children, or -1 if not expanded
            children_i = tree.children_index[batch_index, node_i, a_i]
            if children_i >= 0:
                label, color = node_to_str(
                    node_i=children_i,
                    reward=tree.children_rewards[batch_index, node_i, a_i],
                    discount=tree.children_discounts[batch_index, node_i, a_i])
                graph.add_node(children_i, label=label, color=color)
                graph.add_edge(node_i, children_i, label=edge_to_str(node_i, a_i))

    return graph


if __name__ == '__main__':

    # run ='/mnt/megatron/data/dnd5e/wandb/offline-run-20241124_162033-01psyo9h'
    run = '/mnt/megatron/data/dnd5e/wandb/latest-run'
    run = Path(run)
    epoch = 199
    plot_tree = False
    selected_batch = 0


    with run.joinpath(f'files/data_epoch{epoch:03}.pkl').open('rb') as f:
        checksums, rng_selfplay_devices, dummy_obs, dummy_policy, dummy_value = pickle.load(f)


    n_device, t_max, batch = checksums.observation_checksum.shape
    print(f'devices {n_device} selfplay_batch_size {batch} selfplay_max_num_steps {t_max}')

    # verify the environment
    env = dnd5e.DND5E()
    env = dnd5e.wrap_win_first_death(env)

    # load and verify the model checkpoint
    rng_key = jax.random.PRNGKey(0)
    rng_key, rng_model, rng_env = jax.random.split(rng_key, 3)
    dummy_state = env.init(rng_env)
    regen_obs = flatten_pytree_batched(dummy_state.observation)
    observation_features = dummy_obs.shape[0]
    model = MLP(observation_features, 128, env.num_actions, rngs=nnx.Rngs(rng_model))
    model = train.load_checkpoint(model, run/Path(f'files/model_epoch{epoch:03}'))
    regen_policy, regen_value = nnx.jit(model)(regen_obs)
    assert jnp.allclose(dummy_obs, regen_obs), "environment start state is different, did you initialize the environment the same?"
    assert jnp.allclose(dummy_policy, regen_policy), "output from MLP is different, did you load the correct checkpoint?"
    assert jnp.allclose(dummy_value, regen_value), "output from MLP is different, did you load the correct checkpoint?"


    # use the model and rng_key to regenerate the selfplay epoch
    selfplay = train.make_selfplay(env, batch, t_max, 64, embed_tree=True)
    data = selfplay(model, rng_selfplay_devices)

    assert jnp.allclose(data.checksums.observation_checksum, checksums.observation_checksum)
    assert jnp.allclose(data.checksums.search_tree_action_checksum, checksums.search_tree_action_checksum)

    sample = train.compute_loss_input(data, batch, t_max)
    print(f'sample.obs.shape {sample.obs.shape}')


    # now select a row from the batch, and print it out, plot the game tree for each step if we want
    t_len = 0

    print(f't  stp {'action':<40} ptgt pnet ploss vtgt  vnet  stval')

    for t in range(t_max):
        t_len += 1
        state = jax.tree.map(lambda x : x[0, t, selected_batch], data.state)
        # next_state = jax.tree.map(lambda x: x[0, t, selected_batch], data.next_state)

        action = data.action[0, t, selected_batch]

        is_legal = state.legal_action_mask[action]
        reward = data.reward[0, t, selected_batch]
        discount = data.discount[0, t, selected_batch]
        terminated = data.terminated[0, t, selected_batch]

        policy_tgt = sample.policy_tgt[0, t, selected_batch]
        value_tgt = sample.value_tgt[0, t, selected_batch]
        target_prob = policy_tgt[action]

        obs = sample.obs[0, t, selected_batch]
        network_policy, network_value = model(obs)
        network_value = network_value.squeeze(-1)
        policy_probs = jax.nn.softmax(network_policy)
        policy_prob = policy_probs[action]
        ploss = jnp.mean(optax.softmax_cross_entropy(network_policy, policy_tgt))
        vloss = (network_value - value_tgt) ** 2

        search_tree_summary = jax.tree.map(lambda x: x[0, t, selected_batch], data.search_tree_summary)
        search_tree_value = search_tree_summary.value
        search_tree_qvalue = search_tree_summary.qvalues[action]

        action = dnd5e.decode_action(action, state.current_player, state.scene.party.pos)
        action_repr = dnd5e.repr_action(action)

        print(f'{t:02} {t_len:<3} {action_repr:<40} {target_prob:1.2f} {policy_prob:1.2f} {ploss:1.3f} '
              f'{value_tgt:+1.2f} {network_value:+1.2f} {vloss:1.3f} {search_tree_value:+2.2f} {search_tree_qvalue:+2.2f} '
              f'{reward:+2.2f} {discount:+2.2f} {terminated}'
              )

        if plot_tree:

            search_tree = jax.tree.map(lambda x: x[0, t, 0], data.search_tree[0])
            batched = jax.tree.map(lambda x, y: jnp.stack([x, y]), search_tree, search_tree)

            search_tree_dir = run / Path(f'batch_{selected_batch:03}')
            search_tree_dir.mkdir(exist_ok=True)

            graph = convert_tree_to_graph(batched)
            file_path = search_tree_dir / Path(f'{t:02}_{action_repr}.png'.replace(" ", ''))
            graph.draw(str(file_path), prog='dot')

        if terminated:
            t_len = 0
