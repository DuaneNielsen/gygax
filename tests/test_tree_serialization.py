from functools import partial

import pytest
import jax.numpy as jnp
from jax import numpy as jnp
from jax.numpy import array_equal
from tree_serialization import cum_bins, get_metadata, \
    flatten_pytree_batched, unflatten_pytree_batched
import jax

def test_3d_input():
    # Create a 3D input array of shape (2, 3, 4)
    x = jnp.array([
        [[0, 2, 5, 8],
         [1, 3, 6, 9],
         [2, 4, 7, 10]],
        [[1, 3, 6, 9],
         [2, 4, 7, 10],
         [0, 2, 5, 8]]
    ])
    result = cum_bins(x, lower_bound=0, upper_bound=10, num_bins=5)

    # Expected shape should be (2, 3, 4, 5)
    assert result.shape == (2, 3, 4, 5)

    # Test a specific slice
    expected_slice = jnp.array([1., 1., 1., 0., 0.])  # for value 5
    assert array_equal(result[0, 0, 2], expected_slice)


def test_2d_input():
    x = jnp.array([[0, 5, 10],
                   [2, 6, 8]])
    result = cum_bins(x, lower_bound=0, upper_bound=10, num_bins=4)

    expected = jnp.array([
        [[0., 0., 0., 0.],  # 0
         [1., 1., 0., 0.],  # 5
         [1., 1., 1., 1.]],  # 10
        [[1., 0., 0., 0.],  # 2
         [1., 1., 1., 0.],  # 6
         [1., 1., 1., 1.]]  # 8
    ])
    assert array_equal(result, expected)


def test_1d_input_compatibility():
    # Ensure the function still works with 1D input
    x = jnp.array([0, 5, 10])
    result = cum_bins(x, lower_bound=0, upper_bound=10, num_bins=5)

    expected = jnp.array([
        [0., 0., 0., 0., 0.],  # 0
        [1., 1., 1., 0., 0.],  # 5
        [1., 1., 1., 1., 1.]  # 10
    ])
    assert array_equal(result, expected)


def test_4d_input():
    # Test with 4D input
    x = jnp.ones((2, 2, 2, 3)) * 5  # All values are 5
    result = cum_bins(x, lower_bound=0, upper_bound=10, num_bins=3)

    # Check shape
    assert result.shape == (2, 2, 2, 3, 3)

    # All values should be [1., 1., 0.] since 5 is at the midpoint
    expected_slice = jnp.array([1., 1., 0.])
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(3):
                    assert array_equal(result[i, j, k, l], expected_slice)


def test_empty_leading_dims():
    # Test with empty leading dimensions
    x = jnp.zeros((0, 5))
    result = cum_bins(x, lower_bound=0, upper_bound=10, num_bins=3)
    assert result.shape == (0, 5, 3)


def test_output_dtype():
    x = jnp.array([[[1, 2], [3, 4]]])
    result = cum_bins(x, lower_bound=0, upper_bound=10, num_bins=5)
    assert result.dtype == jnp.float32


def test_input_validation():
    with pytest.raises(ValueError):
        x = jnp.array([[[1, 2], [3, 4]]])
        cum_bins(x, lower_bound=5, upper_bound=5, num_bins=3)

    with pytest.raises(ValueError):
        cum_bins(x, lower_bound=0, upper_bound=10, num_bins=0)


def test_broadcasting():
    # Test that broadcasting works correctly with different shaped inputs
    x = jnp.array([[1], [2], [3]])  # Shape (3, 1)
    result = cum_bins(x, lower_bound=0, upper_bound=10, num_bins=4)
    assert result.shape == (3, 1, 4)



def test_vmapped_flatten_unflatten():
    """Test the vmapped versions of flatten and unflatten operations"""
    # Create a sample nested structure with batch dimension
    batch_size = 3
    sample_data = {
        'a': jnp.ones((batch_size, 2, 3)),
        'b': {
            'c': jnp.zeros((batch_size, 4)),
            'd': jnp.ones((batch_size, 2, 2))
        }
    }

    # Get metadata (constant across batch)
    metadata = get_metadata(sample_data)

    # Create vmapped versions
    vmap_flatten = jax.vmap(flatten_pytree_batched)
    vmap_unflatten = jax.vmap(partial(unflatten_pytree_batched, metadata=metadata))

    # Test flattening and unflattening
    flat_arrays = vmap_flatten(sample_data)
    reconstructed = vmap_unflatten(flat_arrays)

    # Validate shape of flattened arrays
    assert flat_arrays.shape == (batch_size, metadata.total_size), \
        f"Expected shape {(batch_size, metadata.total_size)}, got {flat_arrays.shape}"

    # Validate reconstruction
    for orig_leaf, recon_leaf in zip(
            jax.tree_util.tree_leaves(sample_data),
            jax.tree_util.tree_leaves(reconstructed)
    ):
        assert jnp.array_equal(orig_leaf, recon_leaf), \
            f"Mismatch: original {orig_leaf.shape} vs reconstructed {recon_leaf.shape}"

    print("All tests passed!")
    return flat_arrays, reconstructed
