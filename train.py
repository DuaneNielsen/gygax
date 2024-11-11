import flax.linen as nn
from argparse import ArgumentParser
from typing import Sequence, List
import jax
import jax.numpy as jnp
import dnd5e
from dataclasses import dataclass
import optax
from functools import partial
from flax.training import train_state


class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = nn.relu(x)
        x = nn.Dense(self.features[-1])(x)
        return x

@jax.vmap
def flatten_and_concat_dataclass(data: dataclass) -> jnp.ndarray:
    """
    Flattens and concatenates all fields of a Chex dataclass.

    Args:
        data: A Chex dataclass instance.

    Returns:
        A 1D JAX array containing all flattened and concatenated fields.
    """
    # Flatten the dataclass
    leaves = jax.tree_util.tree_leaves(data)

    # Ensure all leaves are arrays and flatten them
    flat_arrays = [jnp.ravel(leaf) if isinstance(leaf, jnp.ndarray) else jnp.array([leaf]).ravel()
                   for leaf in leaves]

    # Concatenate all flattened arrays
    return jnp.concatenate(flat_arrays)


def target_signal(state):
    N, P, C = state.scene.party.hitpoints.shape
    return jnp.sum(state.scene.party.hitpoints.reshape(N, P * C), -1)


@jax.jit
def train_step(state, observation, target):
    def loss_fn(params):
        predictions = state.apply_fn(params, observation)
        loss = jnp.mean((predictions - target) ** 2)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    return state.apply_gradients(grads=grads), loss


@jax.jit
def run_epoch(state, observation, target):
    batches, batch_size = observation.shape[0:2]
    def body_fn(i, val):
        state, loss_sum = val
        state, loss = train_step(state, observation[i], target[i])
        return state, loss_sum + loss

    return jax.lax.fori_loop(0, batches, body_fn, (state, 0))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--features', type=int, nargs='+', default=[64, 32, 1])
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--dataset_size', type=int, default=8*10)
    args = parser.parse_args()

    # Initialize the model
    rng_key = jax.random.PRNGKey(0)
    rng_key, rng_env, rng_model = jax.random.split(rng_key, 3)

    env = dnd5e.DND5E()
    env_init, env_step = jax.vmap(env.init), jax.vmap(env.step)
    state = env_init(jax.random.split(rng_env, args.dataset_size))
    observation = flatten_and_concat_dataclass(state.observation)
    target = target_signal(state)

    leading_dims = args.dataset_size//args.batch_size, args.batch_size

    observation = observation.reshape(*leading_dims, observation.shape[-1])
    target = target.reshape(*leading_dims)

    print(observation.shape)
    print(target.shape)

    model = MLP(features=args.features)
    params = model.init(rng_model, observation)
    optimizer = optax.adam(args.learning_rate)
    tstate = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    params = model.init(rng_key, observation)

    # Forward pass
    output = model.apply(params, observation)
    print(output.shape)

    for epoch in range(args.num_epochs):
        rng_key, rng_epoch = jax.random.split(rng_key)
        tstate, epoch_loss = run_epoch(tstate, observation, target)
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / 100:.4f}")

        # Final evaluation
    final_state = env_init(jax.random.split(rng_key, args.batch_size))
    final_observation = flatten_and_concat_dataclass(final_state.observation)
    final_target = target_signal(final_state)
    final_prediction = model.apply(tstate.params, final_observation)
    final_loss = jnp.mean((final_prediction - final_target) ** 2)
    print(f"Final evaluation loss: {final_loss:.4f}")
