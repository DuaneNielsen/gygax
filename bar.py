import jax
import jax.numpy as jnp


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
