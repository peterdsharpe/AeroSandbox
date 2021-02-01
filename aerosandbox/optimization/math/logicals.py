import numpy as np
import casadi as cas


def clip(
        x,
        min,
        max
):
    """
    Clip a value to a range.
    Args:
        x: Value to clip.
        min: Minimum value to clip to.
        max: Maximum value to clip to.

    Returns:

    """
    return np.fmin(np.fmax(x, min), max)
