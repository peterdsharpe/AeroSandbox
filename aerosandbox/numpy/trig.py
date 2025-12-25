"""Trigonometric functions with degree-based variants.

This module provides degree-based trigonometric functions as a convenience
wrapper around NumPy's radian-based functions.
"""

import numpy as _onp
from numpy import pi as _pi
from aerosandbox.numpy.typing import Vectorizable, Array

_deg2rad = 180.0 / _pi
_rad2deg = _pi / 180.0


def degrees(x: Vectorizable) -> Array:
    """Convert an angle from radians to degrees.

    Parameters
    ----------
    x : Vectorizable
        Angle in radians.

    Returns
    -------
    Array
        Angle in degrees.

    See Also
    --------
    numpy.degrees : https://numpy.org/doc/stable/reference/generated/numpy.degrees.html
    """
    return x * _deg2rad


def radians(x: Vectorizable) -> Array:
    """Convert an angle from degrees to radians.

    Parameters
    ----------
    x : Vectorizable
        Angle in degrees.

    Returns
    -------
    Array
        Angle in radians.

    See Also
    --------
    numpy.radians : https://numpy.org/doc/stable/reference/generated/numpy.radians.html
    """
    return x * _rad2deg


def sind(x: Vectorizable) -> Array:
    """Compute the sine of an angle given in degrees.

    Parameters
    ----------
    x : Vectorizable
        Angle in degrees.

    Returns
    -------
    Array
        Sine of the angle.
    """
    return _onp.sin(radians(x))


def cosd(x: Vectorizable) -> Array:
    """Compute the cosine of an angle given in degrees.

    Parameters
    ----------
    x : Vectorizable
        Angle in degrees.

    Returns
    -------
    Array
        Cosine of the angle.
    """
    return _onp.cos(radians(x))


def tand(x: Vectorizable) -> Array:
    """Compute the tangent of an angle given in degrees.

    Parameters
    ----------
    x : Vectorizable
        Angle in degrees.

    Returns
    -------
    Array
        Tangent of the angle.
    """
    return _onp.tan(radians(x))


def arcsind(x: Vectorizable) -> Array:
    """Compute the inverse sine, returning result in degrees.

    Parameters
    ----------
    x : Vectorizable
        Value between -1 and 1.

    Returns
    -------
    Array
        Angle in degrees, in the range [-90, 90].
    """
    return degrees(_onp.arcsin(x))


def arccosd(x: Vectorizable) -> Array:
    """Compute the inverse cosine, returning result in degrees.

    Parameters
    ----------
    x : Vectorizable
        Value between -1 and 1.

    Returns
    -------
    Array
        Angle in degrees, in the range [0, 180].
    """
    return degrees(_onp.arccos(x))


def arctan2d(y: Vectorizable, x: Vectorizable) -> Array:
    """Compute the four-quadrant inverse tangent, returning result in degrees.

    Parameters
    ----------
    y : Vectorizable
        y-coordinate.
    x : Vectorizable
        x-coordinate.

    Returns
    -------
    Array
        Angle in degrees, in the range [-180, 180].

    See Also
    --------
    numpy.arctan2 : https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html
    """
    return degrees(_onp.arctan2(y, x))
