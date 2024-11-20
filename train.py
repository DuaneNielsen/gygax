import time

from flax import nnx
from argparse import ArgumentParser
from typing import Sequence, List
import jax
import jax.numpy as jnp
import dnd5e
from dataclasses import dataclass
import optax
from pgx.experimental import act_randomly, auto_reset
from random import shuffle
import pickle
from pathlib import Path
from functools import partial
from tree_serialization import flatten_pytree_batched
from plots import LiveProbabilityPlot
from constants import Actions, N_ACTIONS, N_CHARACTERS, N_PLAYERS
import numpy as onp
from play import PartyVisualizer
import pgx
import mctx
from typing import NamedTuple
from mctx._src.tree import SearchSummary

devices = jax.local_devices()
num_devices = len(devices)

vmap_flatten = jax.vmap(flatten_pytree_batched)


class MLP(nnx.Module):
    def __init__(self, din: int, hidden: int, n_actions: int, rngs: nnx.Rngs):
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


@nnx.jit
def train_step(model, optimizer, observation, target_policy, target_value):
    def loss_fn(model, observation, target_policy, target_value):
        policy, value = model(observation)
        return jnp.mean(optax.softmax_cross_entropy(policy, target_policy) + (value - target_value) ** 2)

    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, observation, target_policy, target_value)

    optimizer.update(grads)
    return loss


def get_recurrent_function(env):
    step = jax.vmap(jax.jit(env.step))

    def recurrent_fn(model, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State):
        # model: params
        # state: embedding
        del rng_key
        current_player = state.current_player.squeeze(-1)
        state = step(state, action)
        observation = vmap_flatten(state.observation)

        logits, value = jax.jit(model)(observation)
        value = value.squeeze(-1)

        # mask invalid actions
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

        reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
        value = jnp.where(state.terminated, 0.0, value)

        # negate the discount when control passes to the opposing party
        tt = state.scene.turn_tracker
        discount = jnp.where(tt.party.squeeze(-1) != tt.prev_party.squeeze(-1), -1.0 * jnp.ones_like(value),
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


class SelfplayOutput(NamedTuple):
    state: dnd5e.State
    obs: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    action_weights: jnp.ndarray
    discount: jnp.ndarray
    search_tree_summary: SearchSummary


def make_selfplay(env, selfplay_batch_size, selfplay_max_steps, selfplay_num_simulations):
    state_axes = nnx.StateAxes({...: None})

    @nnx.pmap(in_axes=(state_axes, 0), out_axes=0, devices=jax.devices())
    def selfplay(model, rng_key: jnp.ndarray) -> SelfplayOutput:
        batch_size = selfplay_batch_size // num_devices
        recurrent_fn = get_recurrent_function(env)

        def step_fn(state, rng_key) -> SelfplayOutput:
            rng_key, rng_search, rng_env = jax.random.split(rng_key, 3)
            observation = vmap_flatten(state.observation)

            logits, value = model(observation)
            value = value.squeeze(-1)
            root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)

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

            # step the environment
            action = jnp.argmax(policy_output.action_weights, -1)
            env_keys = jax.random.split(rng_env, batch_size)
            state = jax.vmap(auto_reset(jax.jit(env.step), jax.jit(env.init)))(state, action, env_keys)

            # calc reward and discount
            current_player = state.current_player
            tt = state.scene.turn_tracker
            reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
            discount = jnp.where(tt.party.squeeze(-1) != tt.prev_party.squeeze(-1),
                                 -1.0 * jnp.ones_like(value),
                                 jnp.ones_like(value))
            discount = jnp.where(state.terminated, 0.0, discount)

            # return the search stats so we can track them
            search_tree_summary = policy_output.search_tree.summary()
            return state, SelfplayOutput(
                state=state,
                obs=observation,
                action_weights=policy_output.action_weights,
                reward=reward,
                terminated=state.terminated,
                discount=discount,
                search_tree_summary=search_tree_summary,
            )

        # init the env and generate a batch of trajectories
        rng_key, rng_env_init = jax.random.split(rng_key, 2)
        state = jax.jit(jax.vmap(env.init))(jax.random.split(rng_env_init, selfplay_batch_size))
        key_seq = jax.random.split(rng_key, selfplay_max_steps)
        _, data = jax.lax.scan(jax.jit(step_fn), state, key_seq)

        return data

    return selfplay


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--features', type=int, nargs='+', default=[64, 32, 1])
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--dataset_size', type=int, default=100)
    parser.add_argument('--selfplay_batch_size', type=int, default=2)
    parser.add_argument('--selfplay_num_simulations', type=int, default=64)
    parser.add_argument('--selfplay_max_steps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # Initialize the random keys
    rng_key = jax.random.PRNGKey(args.seed)
    rng_key, rng_model, rng_env, rng_policy, rng_search = jax.random.split(rng_key, 5)
    env = dnd5e.DND5E()

    state = env.init(rng_env)
    observation_features = flatten_pytree_batched(state.observation).shape[0]
    model = MLP(observation_features, 128, env.num_actions, rngs=nnx.Rngs(rng_model))

    selfplay = make_selfplay(env, args.selfplay_batch_size, args.selfplay_max_steps, args.selfplay_num_simulations)
    rng_selfplay_devices = jax.random.split(rng_search, num_devices)
    data = selfplay(model, rng_selfplay_devices)


