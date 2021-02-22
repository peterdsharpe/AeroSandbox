import numpy as _onp


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
    return _onp.fmin(_onp.fmax(x, min), max)
