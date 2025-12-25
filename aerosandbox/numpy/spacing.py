"""Spacing functions for creating evenly or non-uniformly spaced arrays.

This module provides functions for generating arrays with various spacing
patterns (linear, logarithmic, cosine, sine), working with both NumPy and
CasADi backends.
"""

import numpy as _onp
import casadi as _cas
from aerosandbox.numpy.determine_type import is_casadi_type
from aerosandbox.numpy.typing import Vectorizable, Array


def linspace(
    start: Vectorizable = 0.0,
    stop: Vectorizable = 1.0,
    num: int = 50,
) -> Array:
    """Return evenly spaced numbers over a specified interval.

    Parameters
    ----------
    start : Vectorizable, optional
        The starting value of the sequence. Default is 0.0.
    stop : Vectorizable, optional
        The end value of the sequence. Default is 1.0.
    num : int, optional
        Number of samples to generate. Default is 50.

    Returns
    -------
    Array
        ``num`` evenly spaced samples in the closed interval [start, stop].

    See Also
    --------
    numpy.linspace : https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
    """
    if not is_casadi_type([start, stop, num], recursive=True):
        return _onp.linspace(start, stop, num)
    else:
        return _cas.linspace(start, stop, num)


def cosspace(
    start: Vectorizable = 0.0,
    stop: Vectorizable = 1.0,
    num: int = 50,
) -> Array:
    """Return cosine-spaced numbers over a specified interval.

    Cosine spacing bunches points near both endpoints and is useful for
    polynomial interpolation because these correspond to Chebyshev nodes.

    Parameters
    ----------
    start : Vectorizable, optional
        The starting value of the sequence. Default is 0.0.
    stop : Vectorizable, optional
        The end value of the sequence. Default is 1.0.
    num : int, optional
        Number of samples to generate. Default is 50.

    Returns
    -------
    Array
        ``num`` cosine-spaced samples in the closed interval [start, stop].

    References
    ----------
    .. [1] Chebyshev nodes: https://en.wikipedia.org/wiki/Chebyshev_nodes
    .. [2] Explanation video: https://youtu.be/VSvsVgGbN7I
    """
    mean = (stop + start) / 2
    amp = (stop - start) / 2
    ones = 0 * start + 1
    spaced_array = mean + amp * _onp.cos(linspace(_onp.pi * ones, 0 * ones, num))

    # Fix the endpoints, which might not be exactly right due to floating-point error.
    spaced_array[0] = start
    spaced_array[-1] = stop

    return spaced_array


def sinspace(
    start: Vectorizable = 0.0,
    stop: Vectorizable = 1.0,
    num: int = 50,
    reverse_spacing: bool = False,
) -> Array:
    """Return sine-spaced numbers over a specified interval.

    Sine spacing bunches points near the start of the interval by default.
    This is equivalent to half of a cosine-spaced distribution.

    Parameters
    ----------
    start : Vectorizable, optional
        The starting value of the sequence. Default is 0.0.
    stop : Vectorizable, optional
        The end value of the sequence. Default is 1.0.
    num : int, optional
        Number of samples to generate. Default is 50.
    reverse_spacing : bool, optional
        If True, bunch points near ``stop`` instead of ``start``.
        Default is False.

    Returns
    -------
    Array
        ``num`` sine-spaced samples in the closed interval [start, stop].

    References
    ----------
    .. [1] Explanation video: https://youtu.be/VSvsVgGbN7I
    """
    if reverse_spacing:
        return sinspace(stop, start, num)[::-1]
    ones = 0 * start + 1
    spaced_array = start + (stop - start) * (
        1 - _onp.cos(linspace(0 * ones, _onp.pi / 2 * ones, num))
    )
    # Fix the endpoints, which might not be exactly right due to floating-point error.
    spaced_array[0] = start
    spaced_array[-1] = stop

    return spaced_array


def logspace(
    start: Vectorizable = 0.0,
    stop: Vectorizable = 1.0,
    num: int = 50,
) -> Array:
    """Return numbers spaced evenly on a log scale.

    In linear space, the sequence starts at ``10**start`` and ends at
    ``10**stop``.

    Parameters
    ----------
    start : Vectorizable, optional
        The starting exponent (base 10). Default is 0.0.
    stop : Vectorizable, optional
        The ending exponent (base 10). Default is 1.0.
    num : int, optional
        Number of samples to generate. Default is 50.

    Returns
    -------
    Array
        ``num`` samples, evenly spaced on a log scale.

    See Also
    --------
    numpy.logspace : https://numpy.org/doc/stable/reference/generated/numpy.logspace.html
    """
    if not is_casadi_type([start, stop, num], recursive=True):
        return _onp.logspace(start, stop, num)
    else:
        return 10 ** linspace(start, stop, num)


def geomspace(
    start: Vectorizable = 1.0,
    stop: Vectorizable = 10.0,
    num: int = 50,
) -> Array:
    """Return numbers spaced evenly on a log scale (geometric progression).

    This is similar to ``logspace``, but with endpoints specified directly
    rather than as exponents.

    Parameters
    ----------
    start : Vectorizable, optional
        The starting value of the sequence. Must be positive. Default is 1.0.
    stop : Vectorizable, optional
        The end value of the sequence. Must be positive. Default is 10.0.
    num : int, optional
        Number of samples to generate. Default is 50.

    Returns
    -------
    Array
        ``num`` samples, evenly spaced on a log scale.

    Raises
    ------
    ValueError
        If ``start`` or ``stop`` is not positive.

    See Also
    --------
    numpy.geomspace : https://numpy.org/doc/stable/reference/generated/numpy.geomspace.html
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
