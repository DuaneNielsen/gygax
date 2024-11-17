import jax.numpy as jnp

def create_argmin_mask(array):
    """
    Creates a boolean mask with True at the position of the minimum value in the array.

    Args:
        array: JAX array of any dimension

    Returns:
        Boolean mask of same shape as input array, with True at argmin position
    """
    # Get the flat index of the minimum value
    flat_argmin = jnp.argmin(array)

    # Create a mask of zeros with the same shape
    mask = jnp.zeros_like(array, dtype=bool)

    # Convert flat index to multi-dimensional index
    min_idx = jnp.unravel_index(flat_argmin, array.shape)

    # Set the minimum position to True
    mask = mask.at[min_idx].set(True)

    return mask