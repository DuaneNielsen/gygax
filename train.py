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
        discount = jnp.where(tt.party.squeeze(-1) != tt.prev_party.squeeze(-1), -1.0 * jnp.ones_like(value), jnp.ones_like(value))
        discount = jnp.where(state.terminated, 0.0, discount)

        recurrent_fn_output = mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=logits,
            value=value,
        )
        return recurrent_fn_output, state

    return recurrent_fn


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--features', type=int, nargs='+', default=[64, 32, 1])
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--dataset_size', type=int, default=100)
    args = parser.parse_args()

    # Initialize the random keys
    rng_key = jax.random.PRNGKey(0)
    rng_key, rng_model, rng_env, rng_policy, rng_search = jax.random.split(rng_key, 5)
    env = dnd5e.DND5E()
    recurrent_fn = get_recurrent_function(env)
    num_actions = env.num_actions

    state = jax.vmap(env.init)(jax.random.split(rng_env, args.batch_size))
    observation = vmap_flatten(state.observation)
    model = MLP(observation.shape[1], 128, num_actions, rngs=nnx.Rngs(rng_model))

    logits, value = jax.vmap(model)(observation)
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

    root = mctx.RootFnOutput(prior_logits=logits, value=value.squeeze(-1), embedding=state)


    def root_action_selection_fn(rng_key, tree, node_index):
        action = act_randomly(rng_key, tree.embeddings.legal_action_mask[node_index])
        jax.debug.print('{}', action)
        return action


    def interior_action_selection_fn(rng_key, tree, node_index, depth):
        action = act_randomly(rng_key, tree.embeddings.legal_action_mask[node_index])
        jax.debug.print('{}', action)
        return action


    search_tree = mctx.search(params=model,
                              rng_key=rng_search,
                              root=root,
                              recurrent_fn=recurrent_fn,
                              root_action_selection_fn=root_action_selection_fn,
                              interior_action_selection_fn=interior_action_selection_fn,
                              num_simulations=20,
                              )
