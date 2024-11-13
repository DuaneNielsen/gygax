import time


from flax import nnx
from argparse import ArgumentParser
from typing import Sequence, List
import jax
import jax.numpy as jnp
import dnd5e
from dataclasses import dataclass
import optax
from flax.training import train_state
from pgx.experimental import act_randomly, auto_reset
from random import shuffle
import pickle
from pathlib import Path
from functools import partial
from tree_serialization import flatten_pytree_batched
from plots import LiveProbabilityPlot


vmap_flatten = jax.vmap(flatten_pytree_batched)


# class MLP(nn.Module):
#     features: Sequence[int]
#
#     @nn.compact
#     def __call__(self, x):
#         for feat in self.features[:-1]:
#             x = nn.Dense(feat)(x)
#             x = nn.relu(x)
#         x = nn.Dense(self.features[-1])(x)
#         return x


class Linear(nnx.Module):
    def __init__(self, din, dout, rngs: nnx.Rngs):
        self.linear = nnx.Linear(din, dout, rngs=rngs, use_bias=False)

    def __call__(self, x):
        return self.linear(x)


def target_signal(state):
    N, P, C, S = state.observation.party.hitpoints.shape
    return jnp.sum(state.observation.party.hitpoints.reshape(N, P * C * S), -1)


def generate_data(rng_key, batch_size, dataset_size):
    # generate data
    buffer = []

    env = dnd5e.DND5E()
    # env_init, env_step = jax.vmap(env.init), jax.vmap(env.step)
    rng_key, rng_env = jax.random.split(rng_key)
    state = jax.vmap(env.init)(jax.random.split(rng_env, batch_size))
    observation = vmap_flatten(state.observation.party.hitpoints)
    target = target_signal(state)
    for observation, target in zip(observation, target):
        buffer.append((observation, target))

    for i in range(dataset_size - 1):
        rng_key, rng_policy, rng_env = jax.random.split(rng_key, 3)
        keys = jax.random.split(rng_env, batch_size)
        state = jax.vmap(auto_reset(env.step, env.init))(state, act_randomly(rng_policy, state.legal_action_mask), keys)
        observation = vmap_flatten(state.observation.party.hitpoints)
        target = target_signal(state)
        for observation, target in zip(observation ,target):
           buffer.append((observation, target))
    return buffer


def generate_synthetic_data():
    from tree_serialization import cum_bins
    target = jnp.concatenate([jnp.arange(20) for _ in range(5)])
    observation = cum_bins(target, 19)
    target = target[:, None]
    buffer = []
    for o, t in zip(observation, target):
        buffer.append((o, t))
    return buffer

# def target_signal(state):
#     N, P, C = state.scene.party.hitpoints.shape
#     return jnp.sum(jnp.clip(state.scene.party.hitpoints.reshape(N, P * C), min=0), -1)



@nnx.jit
def train_step(model, optimizer, observation, target):
    def loss_fn(model, observation, target):
        predictions = model(observation)
        return jnp.mean((predictions - target) ** 2)

    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, observation, target)

    optimizer.update(grads)
    # grad_leaves = jax.tree_leaves(grads)
    # grad_mean = jnp.mean(jnp.array([jnp.mean(leaf) for leaf in grad_leaves]))
    # jax.debug.print('grad {}', grad_mean)
    return loss




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--features', type=int, nargs='+', default=[64, 32, 1])
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--num_epochs', type=int, default=600)
    parser.add_argument('--dataset_size', type=int, default=100)
    args = parser.parse_args()

    # Initialize the model
    rng_key = jax.random.PRNGKey(0)
    rng_key, rng_model = jax.random.split(rng_key, 2)
    # model = MLP(args.features)

    dataset_file = Path('buffer.pkl')

    if dataset_file.exists():
        with dataset_file.open('rb') as f:
            buffer = pickle.load(f)
    else:
        # buffer = generate_data(rng_key, args.batch_size, args.dataset_size)
        buffer = generate_synthetic_data()
        with dataset_file.open('wb') as f:
            pickle.dump(buffer, f)

    shuffle(buffer)
    observation, target = tuple([jnp.stack(d) for d in zip(*buffer)])
    # max_target = jnp.max(target)
    # target = target / max_target

    # train test split
    test_split = len(buffer)//10
    train_observation, test_observation = observation[:-test_split], observation[-test_split:]
    train_target, test_target = target[:-test_split], target[-test_split:]

    # split into minibatches
    train_observation = jnp.stack(jnp.split(train_observation, train_observation.shape[0]//args.batch_size))
    train_target = jnp.stack(jnp.split(train_target, train_target.shape[0]//args.batch_size))

    model = Linear(observation.shape[1], 1, nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.sgd(args.learning_rate))
    pred = model(train_observation[0])
    assert pred.shape == train_target[0].shape

    # Create the plot
    plot = LiveProbabilityPlot(num_probabilities=model.linear.kernel.value.shape[0])
    plot.update(model.linear.kernel.value[:, 0])
    # jax.debug.print('{}', dense_0)
    # time.sleep(0.1)

    from statistics import mean

    for epoch in range(args.num_epochs):
        losses = []
        for obs, target in zip(train_observation, train_target):
            loss = train_step(model, optimizer, obs, target)
            losses.append(loss.item())
            plot.update(model.linear.kernel.value[:, 0])
        print(f'loss {mean(losses)}')

    from math import prod
    test_prediction = model(test_observation)
    test_loss = jnp.mean((test_prediction - test_target) ** 2)
    jax.debug.print('{}', test_prediction.squeeze())
    jax.debug.print('{}', test_target)

    print(f"Final evaluation loss: {test_loss:.4f}")
