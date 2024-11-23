from typing import Optional, Sequence, Callable
import pygraphviz
import chex
import mctx
import dnd5e
import jax
import jax.numpy as jnp
from argparse import ArgumentParser
from train import MLP, get_recurrent_function, vmap_flatten
from tree_serialization import flatten_pytree_batched
from flax import nnx
import orbax.checkpoint as ocp
from pathlib import Path


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
        pos = tree.embeddings.scene.party.pos[batch_index, node_i]
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


def make_tree(env, rng_key, selfplay_batch_size, selfplay_num_simulations):
    recurrent_fn = get_recurrent_function(env)
    rng_key, rng_env_init = jax.random.split(rng_key, 2)
    state = jax.jit(jax.vmap(env.init))(jax.random.split(rng_env_init, selfplay_batch_size))
    # key_seq = jax.random.split(rng_key, selfplay_max_steps)
    rng_key, rng_search, rng_env = jax.random.split(rng_key, 3)
    observation = vmap_flatten(state.observation)
    logits, value = model(observation)
    value = value.squeeze(-1)
    root = mctx.RootFnOutput(
        prior_logits=logits,
        value=value,
        embedding=state
    )

    policy_output = mctx.gumbel_muzero_policy(
        params=model,
        rng_key=rng_search,
        root=root,
        recurrent_fn=nnx.jit(recurrent_fn),
        num_simulations=selfplay_num_simulations,
        invalid_actions=~state.legal_action_mask,
        qtransform=mctx.qtransform_completed_by_mix_value,
        gumbel_scale=1.0,
    )

    return policy_output


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--features', type=int, nargs='+', default=[64, 32, 1])
    parser.add_argument('--selfplay_batch_size', type=int, default=2)
    parser.add_argument('--selfplay_num_simulations', type=int, default=256)
    parser.add_argument('--selfplay_max_steps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    rng_key = jax.random.PRNGKey(args.seed)
    rng_key, rng_model, rng_env = jax.random.split(rng_key, 3)

    print('loading env')
    env = dnd5e.DND5E()
    # env = dnd5e.wrap_reward_on_hitbar_percentage(env)
    env = dnd5e.wrap_win_first_death(env)

    print('loading network')
    state = env.init(rng_env)
    observation_features = flatten_pytree_batched(state.observation).shape[0]
    model = MLP(observation_features, 128, env.num_actions, rngs=nnx.Rngs(rng_model))
    abstract_model = nnx.eval_shape(lambda: model)
    graphdef, abstract_state = nnx.split(abstract_model)
    checkpointer = ocp.StandardCheckpointer()
    state_restored = checkpointer.restore(Path('../wandb/latest-run/files/latest').absolute(), abstract_state)
    model = nnx.merge(graphdef, state_restored)

    print('making tree')
    policy_output = make_tree(env, rng_key, args.selfplay_batch_size, args.selfplay_num_simulations)

    print('writing graph image')
    graph = convert_tree_to_graph(policy_output.search_tree)
    graph.draw("search_tree.png", prog="dot")
