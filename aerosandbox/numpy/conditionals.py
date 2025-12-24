"""Conditional functions for the AeroSandbox NumPy-like interface.

This module provides conditional selection functions that work with both
NumPy arrays and CasADi symbolic arrays.
"""
import numpy as _onp
import casadi as _cas
from aerosandbox.numpy.determine_type import is_casadi_type
from aerosandbox.numpy.typing import Array, Vectorizable


def where(
    condition: Array,
    value_if_true: Vectorizable,
    value_if_false: Vectorizable,
) -> Array:
    """Return elements chosen from x or y depending on condition.

    Parameters
    ----------
    condition : Array
        Where True, yield ``value_if_true``, otherwise yield ``value_if_false``.
    value_if_true : Vectorizable
        Values from which to choose where ``condition`` is True.
    value_if_false : Vectorizable
        Values from which to choose where ``condition`` is False.

    Returns
    -------
    Array
        An array with elements from ``value_if_true`` where ``condition`` is
        True, and elements from ``value_if_false`` elsewhere.

    See Also
    --------
    numpy.where : https://numpy.org/doc/stable/reference/generated/numpy.where.html
    """
    if not is_casadi_type([condition, value_if_true, value_if_false], recursive=True):
        return _onp.where(condition, value_if_true, value_if_false)
    else:
        return _cas.if_else(condition, value_if_true, value_if_false)


def maximum(
    x1: Vectorizable,
    x2: Vectorizable,
) -> Array:
    """Compute the element-wise maximum of two arrays.

    Parameters
    ----------
    x1, x2 : Vectorizable
        The arrays to compare. They must be broadcastable to a common shape.

    Returns
    -------
    Array
        The element-wise maximum of ``x1`` and ``x2``.

    Warnings
    --------
    Not differentiable at the crossover point; will cause issues if you try
    to optimize across a crossover. Consider using ``softmax`` for smooth
    optimization.

    See Also
    --------
    numpy.maximum : https://numpy.org/doc/stable/reference/generated/numpy.maximum.html
    softmax : Differentiable alternative for optimization.
    """
    if not is_casadi_type([x1, x2], recursive=True):
        return _onp.maximum(x1, x2)
    else:
        return where(
            x1 >= x2,
            x1,
            x2,
        )


def minimum(
    x1: Vectorizable,
    x2: Vectorizable,
) -> Array:
    """Compute the element-wise minimum of two arrays.

    Parameters
    ----------
    x1, x2 : Vectorizable
        The arrays to compare. They must be broadcastable to a common shape.

    Returns
    -------
    Array
        The element-wise minimum of ``x1`` and ``x2``.

    Warnings
    --------
    Not differentiable at the crossover point; will cause issues if you try
    to optimize across a crossover. Consider using ``softmin`` for smooth
    optimization.

    See Also
    --------
    numpy.minimum : https://numpy.org/doc/stable/reference/generated/numpy.minimum.html
    softmin : Differentiable alternative for optimization.
    """
    if not is_casadi_type([x1, x2], recursive=True):
        return _onp.minimum(x1, x2)
    else:
        return where(
            x1 <= x2,
            x1,
            x2,
        )
