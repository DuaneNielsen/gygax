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


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--features', type=int, nargs='+', default=[64, 32, 1])
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--dataset_size', type=int, default=100)
    args = parser.parse_args()

    # Initialize the random keys
    rng_key = jax.random.PRNGKey(0)
    rng_key, rng_model, rng_env = jax.random.split(rng_key, 3)
    env = dnd5e.DND5E()
    num_actions = env.num_actions

    state = env.init(rng_key)
    observation = vmap_flatten(state.observation)

    model = MLP(observation.shape[1], 128, num_actions, rngs=nnx.Rngs(rng_model))
