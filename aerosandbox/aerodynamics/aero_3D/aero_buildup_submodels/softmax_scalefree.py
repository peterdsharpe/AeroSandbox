import aerosandbox.numpy as np
from aerosandbox.numpy.typing import Vectorizable


def softmax_scalefree(x: list[Vectorizable]) -> Vectorizable:
    """
    Computes a smooth maximum of the elements of `x`, with the smoothing scale ("softness") chosen
    automatically as 1% of the largest element (with a floor of 1e-6 on that largest element).

    Note: this is intentionally NOT the same function as `aerosandbox.numpy.softmax_scalefree`,
    despite the shared name. That function takes `*args` and scales its softness by the L2 norm of
    the inputs; this one takes a single list and scales by the maximum element. The two give
    (slightly) different numerical results, so they cannot be interchanged without changing
    AeroBuildup outputs.

    Args:
        x: A list of values to take the smooth maximum of.

    Returns: The smooth maximum of the values in `x`.
    """
    if len(x) == 1:
        return x[0]
    else:
        softness = np.max(np.array([1e-6] + x)) * 0.01

        return np.softmax(*x, softness=softness)
