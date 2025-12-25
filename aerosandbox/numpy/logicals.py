"""Logical and boolean operations for the AeroSandbox NumPy-like interface.

This module provides logical operations that work with both NumPy arrays
and CasADi symbolic arrays.
"""

import numpy as _onp
import casadi as _cas
from aerosandbox.numpy.determine_type import is_casadi_type
from aerosandbox.numpy.typing import ArrayLike, Array, Vectorizable


def clip(x: ArrayLike, min: Vectorizable, max: Vectorizable) -> Array:
    """Clip (limit) the values in an array.

    Given an interval, values outside the interval are clipped to the
    interval edges.

    Parameters
    ----------
    x : ArrayLike
        Array containing elements to clip.
    min : Vectorizable
        Minimum value. Elements smaller than this are replaced with ``min``.
    max : Vectorizable
        Maximum value. Elements larger than this are replaced with ``max``.

    Returns
    -------
    Array
        An array with elements of ``x``, but where values < ``min`` are
        replaced with ``min``, and values > ``max`` are replaced with ``max``.

    See Also
    --------
    numpy.clip : https://numpy.org/doc/stable/reference/generated/numpy.clip.html
    """
    return _onp.fmin(_onp.fmax(x, min), max)


def logical_and(x1: ArrayLike, x2: ArrayLike) -> Array:
    """Compute the truth value of x1 AND x2 element-wise.

    Parameters
    ----------
    x1, x2 : ArrayLike
        Input arrays. They must be broadcastable to a common shape.

    Returns
    -------
    Array
        Boolean array of the element-wise logical AND of ``x1`` and ``x2``.

    See Also
    --------
    numpy.logical_and : https://numpy.org/doc/stable/reference/generated/numpy.logical_and.html
    """
    if not is_casadi_type([x1, x2], recursive=True):
        return _onp.logical_and(x1, x2)

    else:
        return _cas.logic_and(x1, x2)


def logical_or(x1: ArrayLike, x2: ArrayLike) -> Array:
    """Compute the truth value of x1 OR x2 element-wise.

    Parameters
    ----------
    x1, x2 : ArrayLike
        Input arrays. They must be broadcastable to a common shape.

    Returns
    -------
    Array
        Boolean array of the element-wise logical OR of ``x1`` and ``x2``.

    See Also
    --------
    numpy.logical_or : https://numpy.org/doc/stable/reference/generated/numpy.logical_or.html
    """
    if not is_casadi_type([x1, x2], recursive=True):
        return _onp.logical_or(x1, x2)

    else:
        return _cas.logic_or(x1, x2)


def logical_not(x: ArrayLike) -> Array:
    """Compute the truth value of NOT x element-wise.

    Parameters
    ----------
    x : ArrayLike
        Input array.

    Returns
    -------
    Array
        Boolean array of the element-wise logical NOT of ``x``.

    See Also
    --------
    numpy.logical_not : https://numpy.org/doc/stable/reference/generated/numpy.logical_not.html
    """
    if not is_casadi_type(x, recursive=False):
        return _onp.logical_not(x)

    else:
        return _cas.logic_not(x)


def all(a: ArrayLike) -> bool:  # TODO add axis functionality
    """Test whether all array elements evaluate to True.

    Parameters
    ----------
    a : ArrayLike
        Input array.

    Returns
    -------
    bool
        True if all elements evaluate to True, False otherwise.

    See Also
    --------
    numpy.all : https://numpy.org/doc/stable/reference/generated/numpy.all.html
    """
    if not is_casadi_type(a, recursive=False):
        return _onp.all(a)

    else:
        try:
            return _cas.logic_all(a)
        except NotImplementedError:
            return False


def any(a: ArrayLike) -> bool:  # TODO add axis functionality
    """Test whether any array element evaluates to True.

    Parameters
    ----------
    a : ArrayLike
        Input array.

    Returns
    -------
    bool
        True if any element evaluates to True, False otherwise.

    See Also
    --------
    numpy.any : https://numpy.org/doc/stable/reference/generated/numpy.any.html
    """
    if not is_casadi_type(a, recursive=False):
        return _onp.any(a)

    else:
        try:
            return _cas.logic_any(a)
        except NotImplementedError:
            return False
