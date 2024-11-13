import jax
import jax.numpy as jnp
from typing import Tuple, NamedTuple, List, TypeVar

T = TypeVar('T')  # Type variable for the pytree


def cum_bins(x, upper_bound, lower_bound=0, num_bins=None):
    """
    Create cumulative binary bins for each value in the input array with arbitrary leading dimensions.

    Args:
        x: array of integers with shape (..., I) where ... represents arbitrary leading dimensions
        lower_bound: scalar, minimum value for rescaling
        upper_bound: scalar, maximum value for rescaling
        num_bins: scalar, number of bins to create

    Returns:
        array of shape (..., I, num_bins) where each final slice contains cumulative binary values

    Raises:
        ValueError: If num_bins <= 0 or if lower_bound equals upper_bound
    """
    if num_bins is None:
        num_bins = upper_bound - lower_bound
    if num_bins <= 0:
        raise ValueError("Number of bins must be positive")
    if lower_bound == upper_bound:
        raise ValueError("Lower bound cannot equal upper_bound")

    # Rescale the input array to [0, 1]
    x_scaled = (x - lower_bound) / (upper_bound - lower_bound)

    # Create bin edges (including both endpoints)
    bin_edges = jnp.linspace(0, 1, num_bins + 1)

    # Reshape x_scaled to add a new axis for bins at the end
    x_expanded = jnp.expand_dims(x_scaled, axis=-1)

    # Compare x_scaled with each bin edge
    # Broadcasting will handle all leading dimensions automatically
    comparisons = x_expanded > bin_edges

    # Remove the last comparison and convert to float32
    binned_array = comparisons[..., :-1].astype(jnp.float32)

    return binned_array




class FlattenMetadata(NamedTuple):
    """Metadata required to reconstruct the original structure"""
    treedef: jax.tree_util.PyTreeDef
    shapes: List[Tuple[int, ...]]
    total_size: int  # total size of flattened array


def get_metadata(pytree: T) -> FlattenMetadata:
    """
    Get metadata for a pytree without processing any arrays.
    Since metadata is constant across batch dimension, we only need to compute it once.
    """
    leaves = jax.tree_util.tree_leaves(pytree)
    treedef = jax.tree_util.tree_structure(pytree)

    # Get shapes excluding batch dimension
    shapes = [leaf.shape[1:] for leaf in leaves]

    # Calculate total size of flattened array (excluding batch dim)
    total_size = sum(int(jnp.prod(jnp.array(shape))) for shape in shapes)

    return FlattenMetadata(treedef, shapes, total_size)


def flatten_pytree_batched(pytree: T) -> jnp.ndarray:
    """
    Vmappable version of flatten_pytree that only processes arrays.
    Metadata handling is separated out since it's constant across batch dimension.

    Args:
        pytree: Any pytree structure containing float32 arrays

    Returns:
        1D float32 array containing all values
    """
    leaves = jax.tree_util.tree_leaves(pytree)
    flat_arrays = [leaf.reshape((-1,)) for leaf in leaves]
    return jnp.concatenate(flat_arrays).astype(jnp.float32)


def unflatten_pytree_batched(flat_array: jnp.ndarray, metadata: FlattenMetadata) -> T:
    """
    Vmappable version of unflatten_pytree.

    Args:
        flat_array: 1D float32 array containing all values
        metadata: FlattenMetadata containing reconstruction information

    Returns:
        Reconstructed pytree with original structure
    """
    sizes = [int(jnp.prod(jnp.array(shape))) for shape in metadata.shapes]
    splits = jnp.cumsum(jnp.array([0] + sizes))

    arrays = [
        flat_array[splits[i]:splits[i + 1]].reshape(metadata.shapes[i])
        for i in range(len(metadata.shapes))
    ]

    return jax.tree_util.tree_unflatten(metadata.treedef, arrays)
