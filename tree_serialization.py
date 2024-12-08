import jax
import jax.numpy as jnp
from typing import Tuple, NamedTuple, List, TypeVar, Container

from jax import numpy, nn as nn

T = TypeVar('T')  # Type variable for the pytree


def cum_bins(x, upper_bound, lower_bound=0, num_bins=None, dtype=jnp.float32):
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
    binned_array = comparisons[..., :-1].astype(dtype)

    return binned_array


class CumBinType(type):
    """
    CumBinType can be used as type hints to help automatically convert
    state variables to observations as cumulative bins

    example:

    ExampleCumBin = CumBinType('ExampleCumBin', (), {}, upper=10, lower=0, n_bins=5)

    """
    def __new__(cls, name, bases, attrs, *, upper, lower=0, n_bins=None):
        attrs['_upper'] = upper
        attrs['_lower'] = lower
        # If n_bins not provided, calculate from bounds
        attrs['_n_bins'] = n_bins if n_bins is not None else (upper - lower + 1)
        return super().__new__(cls, name, bases, attrs)

    def __instancecheck__(cls, instance):
        return (isinstance(instance, jnp.ndarray) and
                instance.size == cls._n_bins)

    def __subclasscheck__(cls, subclass):
        return isinstance(subclass, CumBinType)

    @property
    def upper(cls):
        return cls._upper

    @property
    def lower(cls):
        return cls._lower

    @property
    def n_bins(cls):
        return cls._n_bins


class OneHotType(type):
    def __new__(cls, name, bases, attrs, *, n_clessas):
        # If n_bins not provided, calculate from bounds
        attrs['_n_classes'] = n_clessas
        return super().__new__(cls, name, bases, attrs)

    @property
    def n_classes(cls):
        return cls._n_classes

    def __subclasscheck__(cls, subclass):
        return isinstance(subclass, CumBinType)


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


def convert_to_observation(character, clazz: T, dtype=jnp.float32) -> T:

    kwargs = {}

    for field_name, field_info in clazz.__dataclass_fields__.items():

        if issubclass(field_info.type, Container):
            field_value = getattr(character, field_name)
            kwargs[field_name] = convert_to_observation(field_value, field_info.type)

        elif isinstance(field_info.type, CumBinType):
            field_value = getattr(character, field_name)
            u, l, n = field_info.type.upper, field_info.type.lower, field_info.type.n_bins
            kwargs[field_name] = cum_bins(field_value, upper_bound=u, lower_bound=l, num_bins=n, dtype=dtype)

        elif isinstance(field_info.type, OneHotType):
            field_value = getattr(character, field_name)
            kwargs[field_name] = nn.one_hot(field_value, field_info.type.n_classes, dtype=dtype)

        elif issubclass(field_info.type, bool) or issubclass(field_info.type, jnp.bool):
            field_value = getattr(character, field_name)
            kwargs[field_name] = jnp.array([field_value], dtype=dtype)

    return clazz(**kwargs)
