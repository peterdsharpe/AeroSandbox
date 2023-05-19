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
    ones = 0 * start + 1
    spaced_array = mean + amp * _onp.cos(
        linspace(
            _onp.pi * ones,
            0 * ones,
            num
        )
    )

    # Fix the endpoints, which might not be exactly right due to floating-point error.
    spaced_array[0] = start
    spaced_array[-1] = stop

    return spaced_array


def sinspace(
        start: float = 0.,
        stop: float = 1.,
        num: int = 50,
        reverse_spacing: bool = False,
):
    """
    Makes a sine-spaced vector. By default, bunches points near the start.

    Sine spacing is exactly identical to half of a cosine-spaced distrubution, in terms of relative separations.

    To learn more about sine spacing and cosine spacing, see this: https://youtu.be/VSvsVgGbN7I

    Args:

        start: Value to start at.

        end: Value to end at.

        num: Number of points in the vector.

        reverse_spacing: Does negative-sine spacing. In other words, if this is True, the points will be bunched near
        the `stop` rather than at the `start`.

    Points are bunched up near the `start` of the interval by default. To reverse this, use the `reverse_spacing`
    parameter.
    """
    if reverse_spacing:
        return sinspace(stop, start, num)[::-1]
    ones = 0 * start + 1
    spaced_array = (
            start + (stop - start) * (1 - _onp.cos(linspace(
        0 * ones,
        _onp.pi / 2 * ones,
        num
    ))
                                      )
    )
    # Fix the endpoints, which might not be exactly right due to floating-point error.
    spaced_array[0] = start
    spaced_array[-1] = stop

    return spaced_array


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
        spaced_array = _onp.log10(10 ** linspace(start, stop, num))

        # Fix the endpoints, which might not be exactly right due to floating-point error.
        spaced_array[0] = start
        spaced_array[-1] = stop

        return spaced_array
