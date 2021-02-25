import numpy as _onp
import casadi as _cas
from aerosandbox.numpy.determine_type import is_casadi_type


def linspace(
        start: float = 0.,
        stop: float = 1.,
        num: int = 50
):
    """
    Returns evenly spaced numbers over a specified interval.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
    """
    if not is_casadi_type([start, stop, num], recursive=True):
        return _onp.linspace(start, stop, num)
    else:
        return _cas.linspace(start, stop, num)


def cosspace(
        start: float = 0.,
        stop: float = 1.,
        num: int = 50
):
    """
    Makes a cosine-spaced vector.

    Cosine spacing is useful because these correspond to Chebyshev nodes: https://en.wikipedia.org/wiki/Chebyshev_nodes

    To learn more about cosine spacing, see this: https://youtu.be/VSvsVgGbN7I

    Args:
        start: Value to start at.
        end: Value to end at.
        num: Number of points in the vector.
    """
    mean = (stop + start) / 2
    amp = (stop - start) / 2
    return mean + amp * _onp.cos(linspace(_onp.pi, 0, num))


def logspace(
        start: float = 0.,
        stop: float = 1.,
        num: int = 50
):
    """
    Return numbers spaced evenly on a log scale.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.logspace.html
    """
    if not is_casadi_type([start, stop, num], recursive=True):
        return _onp.logspace(start, stop, num)
    else:
        return 10 ** linspace(start, stop, num)


def geomspace(
        start: float = 1.,
        stop: float = 10.,
        num: int = 50
):
    """
    Return numbers spaced evenly on a log scale (a geometric progression).

    This is similar to logspace, but with endpoints specified directly.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.geomspace.html
    """
    if not is_casadi_type([start, stop, num], recursive=True):
        return _onp.geomspace(start, stop, num)
    else:
        if start <= 0 or stop <= 0:
            raise ValueError("Both start and stop must be positive!")
        return _onp.log10(10 ** linspace(start, stop, num))
