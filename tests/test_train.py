import numpy as np
import pytest
import jax.numpy as jnp


def compute_trajectory_lengths(terminal, axis=-1):
    # Create an array of indices
    terminal = jnp.moveaxis(terminal, axis, -1)
    shape = [1] * len(terminal.shape[:-1]) + [terminal.shape[-1]]
    indices = jnp.arange(terminal.shape[-1]).reshape(*shape)
    indices = indices * jnp.ones(terminal.shape)

    trajectory_segs = indices[terminal].reshape(terminal.shape)
    lengths = jnp.diff(trajectory_segs, prepend=-1, axis=-1) - 1

    return lengths


def test_traj_len_simple():
    terminated = jnp.array([0, 1, 0, 0, 1, 0, 0, 0, 1], dtype=jnp.bool)
    terminated = terminated.reshape(1, 9, 1)
    lengths = compute_trajectory_lengths(terminated, 1)
    assert jnp.allclose(jnp.array([1, 2, 3]), lengths)


def test_traj_len_multi():
    terminated = jnp.array([
        [0, 1, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 1, 0, 0, 0, 1],
    ], dtype=jnp.bool)
    terminated = terminated.reshape(1, 2, 9)
    lengths = compute_trajectory_lengths(terminated, -1)
    assert jnp.allclose(jnp.array([
        [1, 2, 3],
        [2, 1, 3]
    ]), lengths)
