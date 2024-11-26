import time

from flax import nnx
from argparse import ArgumentParser
from typing import Sequence, List
import jax
import jax.numpy as jnp
import dnd5e
import optax
from pgx.experimental import act_randomly, auto_reset
import pickle
from pathlib import Path
from functools import partial
from tree_serialization import flatten_pytree_batched
import pgx
import mctx
from typing import NamedTuple, TypeVar
from mctx._src.tree import SearchSummary
from mctx._src import tree as tree_lib
import orbax.checkpoint as ocp
import tqdm
import wandb
import chex

devices = jax.local_devices()
num_devices = len(devices)

vmap_flatten = jax.vmap(flatten_pytree_batched)
T = TypeVar("T")


class MLP(nnx.Module):
    def __init__(self, din: int, hidden: int, n_actions: int, rngs: nnx.Rngs):
        super().__init__()
        self.linear_0 = nnx.Linear(din, hidden, rngs=rngs)
        self.linear_1 = nnx.Linear(hidden, hidden, rngs=rngs)
        self.value_head = nnx.Linear(hidden, 1, rngs=rngs, use_bias=False)
        self.policy_head = nnx.Linear(hidden, n_actions, rngs=rngs, use_bias=False)

    def __call__(self, x):
        hidden = self.linear_0(x)
        hidden = nnx.relu(hidden)
        hidden = self.linear_1(hidden)
        hidden = nnx.relu(hidden)
        policy = self.policy_head(hidden)
        value = self.value_head(hidden)
        return policy, value


class MaxEntropyNetwork(nnx.Module):
    """
    A simple network which provides no guidance to the alphazero algorithm,
    action selection will be based entirely on balanced search of the game tree

    Useful as a baseline to verify that the policy and value network is learning something meaningful
    """
    def __init__(self, n_actions):
        super().__init__()
        self.n_actions = n_actions

    def __call__(self, obs):
        batch_size = obs.shape[0]
        policy = jnp.ones((batch_size, self.n_actions)) / self.n_actions
        value = jnp.zeros((batch_size, 1))
        return policy, value


class ModelEnsemble(nnx.Module):
    def __init__(self, models):
        super().__init__()
        # Create multiple models using different PRNGKey splits
        self.models = models

    def __call__(self, x):
        # Run input through all models
        outputs = [model(x) for model in self.models]
        return tuple([jnp.stack(x) for x in list(zip(*outputs))])


def save_checkpoint(model, checkpoint_path):
    _, state = nnx.split(model)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(checkpoint_path, state)


def load_checkpoint(model, checkpoint_path):
    abstract_model = nnx.eval_shape(lambda: model)
    graphdef, abstract_state = nnx.split(abstract_model)
    checkpointer = ocp.StandardCheckpointer()
    state_restored = checkpointer.restore(Path(checkpoint_path).absolute(), abstract_state)
    return nnx.merge(graphdef, state_restored)


@nnx.jit
def train_step(model, optimizer, observation, target_policy, target_value):
    def loss_fn(model, observation, target_policy, target_value):
        policy, value = model(observation)
        return jnp.mean(optax.softmax_cross_entropy(policy, target_policy) + (value - target_value) ** 2)

    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, observation, target_policy, target_value)

    optimizer.update(grads)
    return loss


def get_recurrent_function(env, evaluation=False):
    step = jax.vmap(jax.jit(env.step))

    def recurrent_fn(params, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State):
        if evaluation:
            model, baseline_player = params
        else:
            model = params

        # state: embedding
        del rng_key
        current_player = state.current_player.squeeze(-1)
        state = step(state, action)
        observation = vmap_flatten(state.observation)

        # model
        logits, value = nnx.jit(model)(observation)

        if evaluation:
            batch_range = jnp.arange(logits.shape[1])
            logits = logits[baseline_player, batch_range]
            value = value[baseline_player, batch_range]

        value = value.squeeze(-1)

        # mask invalid actions
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

        reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
        value = jnp.where(state.terminated, 0.0, value)

        # negate the discount when control passes to the opposing party
        tt = state.scene.turn_tracker
        discount = jnp.where(tt.party.squeeze(-1) != tt.prev_party.squeeze(-1),
                             -1.0 * jnp.ones_like(value),
                             jnp.ones_like(value))
        discount = jnp.where(state.terminated, 0.0, discount)

        recurrent_fn_output = mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=logits,
            value=value,
        )
        return recurrent_fn_output, state

    return recurrent_fn

@chex.dataclass
class Checksums:
    observation_checksum: jax.Array
    search_tree_action_checksum: jax.Array


class SelfplayOutput(NamedTuple):
    state: dnd5e.State
    action: jnp.ndarray
    obs: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    action_weights: jnp.ndarray
    discount: jnp.ndarray
    search_tree_summary: SearchSummary
    search_tree: tree_lib.Tree[T]
    checksums: Checksums  # information that can be used to verify search tree regeneration


@chex.dataclass
class Embeddings:
    terminated: jax.Array
    current_player: jax.Array
    pos: jax.Array


def copy_tree_without_embeddings(tree: tree_lib.Tree[T]) -> tree_lib.Tree[T]:
    """Creates a copy of a Tree instance without copying the embeddings field.

    Args:
        tree: The source Tree instance to copy.

    Returns:
        A new Tree instance with copied arrays for all fields except embeddings,
        which maintains a reference to the original embeddings.
    """

    embeddings = Embeddings(
        terminated=jnp.array(tree.embeddings.terminated),
        current_player=jnp.array(tree.embeddings.current_player),
        pos=jnp.array(tree.embeddings.scene.party.pos)
    )

    return tree_lib.Tree(
        node_visits=jnp.array(tree.node_visits),
        raw_values=jnp.array(tree.raw_values),
        node_values=jnp.array(tree.node_values),
        parents=jnp.array(tree.parents),
        action_from_parent=jnp.array(tree.action_from_parent),
        children_index=jnp.array(tree.children_index),
        children_prior_logits=jnp.array(tree.children_prior_logits),
        children_visits=jnp.array(tree.children_visits),
        children_rewards=jnp.array(tree.children_rewards),
        children_discounts=jnp.array(tree.children_discounts),
        children_values=jnp.array(tree.children_values),
        embeddings=embeddings,
        root_invalid_actions=jnp.array(tree.root_invalid_actions),
        extra_data=tree.extra_data
    )


def make_selfplay(env, selfplay_batch_size, selfplay_max_steps, selfplay_num_simulations, embed_tree=False, evaluate=False):
    state_axes = nnx.StateAxes({...: None})

    @nnx.pmap(in_axes=(state_axes, 0), out_axes=0, devices=jax.devices())
    def selfplay(model, selfplay_rng_key: jnp.ndarray) -> SelfplayOutput:
        batch_size = selfplay_batch_size // num_devices
        recurrent_fn = get_recurrent_function(env, evaluate)

        def step_fn(state, rng_key) -> SelfplayOutput:
            rng_key, rng_search, rng_env, rng_policy = jax.random.split(rng_key, 4)
            observation = vmap_flatten(state.observation)
            current_player = state.current_player.squeeze(-1)

            logits, value = nnx.jit(model)(observation)

            if evaluate:
                batch_range = jnp.arange(logits.shape[1])
                logits = logits[current_player, batch_range]
                value = value[current_player, batch_range]

            value = value.squeeze(-1)
            root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)

            # when evaluating, use the baseline
            params = (model, current_player) if evaluate else model

            policy_output = mctx.gumbel_muzero_policy(
                params=params,
                rng_key=rng_search,
                root=root,
                recurrent_fn=nnx.jit(recurrent_fn),
                num_simulations=selfplay_num_simulations,
                invalid_actions=~state.legal_action_mask,
                qtransform=mctx.qtransform_completed_by_mix_value,
                gumbel_scale=1.0,
            )

            # step the environment
            # jax.debug.print('illegal actions {}', state.legal_action_mask[jnp.arange(256), ~policy_output.action].sum())
            action = jax.random.categorical(rng_policy, jnp.log(policy_output.action_weights), axis=-1)
            # jax.debug.print('{} {} {}', rng_policy, action, value)
            env_keys = jax.random.split(rng_env, batch_size)
            next_state = jax.vmap(auto_reset(jax.jit(env.step), jax.jit(env.init)))(state, action, env_keys)

            # calc reward and discount
            tt = next_state.scene.turn_tracker
            reward = next_state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
            discount = jnp.where(tt.party.squeeze(-1) != tt.prev_party.squeeze(-1),
                                 -1.0 * jnp.ones_like(value),
                                 jnp.ones_like(value))
            discount = jnp.where(next_state.terminated, 0.0, discount)

            # return the search stats so we can track them
            search_tree_summary = policy_output.search_tree.summary()

            # used to regenerate the selfplay search for debugging
            action_csum = policy_output.search_tree.action_from_parent.sum(-1)
            observation_csum = observation.sum(-1)
            checksums = Checksums(
                observation_checksum=observation_csum,
                search_tree_action_checksum=action_csum,
            )

            if embed_tree:
                search_tree = copy_tree_without_embeddings(policy_output.search_tree),
            else:
                search_tree = None

            return next_state, SelfplayOutput(
                state=state,
                action=action,
                obs=observation,
                action_weights=policy_output.action_weights,
                reward=reward,
                terminated=next_state.terminated,
                discount=discount,
                search_tree_summary=search_tree_summary,
                search_tree=search_tree,
                checksums=checksums,
            )

        # init the env and generate a batch of trajectories
        rng_key, rng_env_init = jax.random.split(selfplay_rng_key, 2)
        state = jax.jit(jax.vmap(env.init))(jax.random.split(rng_env_init, selfplay_batch_size))
        key_seq = jax.random.split(rng_key, selfplay_max_steps)
        _, data = jax.lax.scan(jax.jit(step_fn), state, key_seq)

        return data

    return selfplay


class Sample(NamedTuple):
    obs: jnp.ndarray
    policy_tgt: jnp.ndarray
    value_tgt: jnp.ndarray
    mask: jnp.ndarray


@partial(jax.pmap, static_broadcasted_argnums=(1, 2))
def compute_loss_input(data: SelfplayOutput, selfplay_batch_size, selfplay_max_num_steps) -> Sample:
    batch_size = selfplay_batch_size // num_devices
    # If episode is truncated, there is no value target
    # So when we compute value loss, we need to mask it
    value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1

    # est_traj_len = jnp.argmin(jnp.arange(value_mask.shape[-1]) * value_mask, -1)

    # Compute value target
    def body_fn(carry, i):
        ix = selfplay_max_num_steps - i - 1
        v = data.reward[ix] + data.discount[ix] * carry
        return v, v

    _, value_tgt = jax.lax.scan(
        body_fn,
        jnp.zeros(batch_size),
        jnp.arange(selfplay_max_num_steps),
    )
    value_tgt = value_tgt[::-1, :]

    return Sample(
        obs=data.obs,
        policy_tgt=data.action_weights,
        value_tgt=value_tgt,
        mask=value_mask,
    )


def write_metrics(data, step, prefix='eval'):
    player_0_wins = (data.state.rewards[..., 0] == 1.).sum()
    player_1_wins = (data.state.rewards[..., 1] == 1.).sum()
    t_len = data.state._step_count.squeeze(-1)[data.terminated]

    wandb.log({
        f"{prefix}_player0_wins": player_0_wins,
        f"{prefix}_player1_wins": player_1_wins,
        f"{prefix}_player0_win_rate": player_0_wins / (player_0_wins + player_1_wins),
        f"{prefix}_traj_len_mean": jnp.mean(t_len),
        f"{prefix}_traj_count": t_len.shape[0]
    }, step=step)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[64, 32, 1])
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--selfplay_batch_size', type=int, default=256)
    parser.add_argument('--selfplay_num_simulations', type=int, default=64)
    parser.add_argument('--selfplay_max_steps', type=int, default=24)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--training_batch_size', type=int, default=8)
    args = parser.parse_args()

    # random keys
    rng_key = jax.random.PRNGKey(args.seed)
    rng_key, rng_model, rng_env, rng_policy, rng_search = jax.random.split(rng_key, 5)

    # environment
    env = dnd5e.DND5E()
    # env = dnd5e.wrap_reward_on_hitbar_percentage(env)
    env = dnd5e.wrap_win_first_death(env)
    # env = dnd5e.wrap_party_initiative(env, 0, 1)

    # setup model
    dummy_state = env.init(rng_env)
    dummy_obs = flatten_pytree_batched(dummy_state.observation)
    observation_features = dummy_obs.shape[0]
    model = MLP(observation_features, 128, env.num_actions, rngs=nnx.Rngs(rng_model))
    optimizer = nnx.Optimizer(model, optax.adam(args.learning_rate))

    # baseline_model = MLP(observation_features, 128, env.num_actions, rngs=nnx.Rngs(rng_model))
    # baseline_chkpt = '/mnt/megatron/data/dnd5e/wandb/offline-run-20241124_222111-6p2bjtub/files/model_epoch199'
    # baseline_model = load_checkpoint(baseline_model, baseline_chkpt)

    baseline_model = MaxEntropyNetwork(env.num_actions)
    baseline_ensemble = ModelEnsemble([model, baseline_model])

    # monitoring
    wandb.init(project=f'alphazero-dnd5e', config=args.__dict__, settings=wandb.Settings(code_dir="."))
    run_dir = Path(f'{wandb.run.dir}')
    frames = 0

    # instantiate selfplay and eval
    selfplay = make_selfplay(env, args.selfplay_batch_size, args.selfplay_max_steps, args.selfplay_num_simulations)
    playbaseline = make_selfplay(env, args.selfplay_batch_size, args.selfplay_max_steps, args.selfplay_num_simulations, evaluate=True)

    for epoch in tqdm.trange(args.num_epochs):

        checkpoint_path = Path(run_dir.absolute() / f'model_epoch{epoch:03}')
        save_checkpoint(model, str(checkpoint_path))
        dummy_policy, dummy_value = nnx.jit(model)(dummy_obs)

        # selfplay
        rng_selfplay_devices = jax.random.split(rng_search, num_devices)
        data = selfplay(model, rng_selfplay_devices)
        # samples > (#devices, batch, max_num_steps, ...)
        with Path(run_dir / f'data_epoch{epoch:03}.pkl').open(mode='wb') as f:
            pickle.dump((data.checksums, rng_selfplay_devices, dummy_obs, dummy_policy, dummy_value), f)

        samples = compute_loss_input(data, args.selfplay_batch_size, args.selfplay_max_steps)

        # load and shuffle the batch
        samples = jax.device_get(samples)  # read into host memory
        samples = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[3:])), samples)
        rng_key, rng_batch = jax.random.split(rng_key)
        ixs = jax.random.permutation(rng_batch, jnp.arange(samples.obs.shape[0]))
        samples = jax.tree_util.tree_map(lambda x: x[ixs], samples)
        num_updates = samples.obs.shape[0] // args.training_batch_size
        minibatches = jax.tree_util.tree_map(
            lambda x: x.reshape((num_updates, num_devices, -1) + x.shape[1:]), samples
        )
        write_metrics(data, step=epoch, prefix='splay')

        # train the network
        loss = jnp.array([4.])
        for i in range(num_updates):
            minibatch: Sample = jax.tree_util.tree_map(lambda x: x[i], minibatches)
            batch_loss = train_step(model, optimizer, minibatch.obs, minibatch.policy_tgt, minibatch.value_tgt)
            loss = batch_loss * 0.5 + loss * 0.5

        wandb.log({'loss': loss.item()}, step=epoch)

        # evaluate the network
        rng_play_baseline = jax.random.fold_in(rng_search, jnp.array(epoch))
        rng_play_baseline_device = jax.random.split(rng_play_baseline, num_devices)
        eval_data = playbaseline(baseline_ensemble, rng_play_baseline_device)
        with Path(run_dir / f'data_epoch{epoch:03}_eval.pkl').open(mode='wb') as f:
            pickle.dump((eval_data.checksums, rng_selfplay_devices, dummy_obs, dummy_policy, dummy_value), f)
        write_metrics(eval_data, step=epoch, prefix='eval')

    # checkpoint
    checkpoint_path = Path(run_dir.absolute() / f'final_epoch{epoch:03}')
    save_checkpoint(model, str(checkpoint_path))
    latest_dir = run_dir.absolute() / f'latest'
    latest_dir.unlink(missing_ok=True)
    latest_dir.symlink_to(checkpoint_path)
