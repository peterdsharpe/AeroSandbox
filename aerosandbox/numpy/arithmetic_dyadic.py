"""Dyadic (two-operand) arithmetic functions for AeroSandbox.

This module provides element-wise binary arithmetic operations that work
with both NumPy arrays and CasADi symbolic arrays, with proper broadcasting.
"""

import numpy as _onp
import casadi as _cas
from aerosandbox.numpy.conditionals import where

from aerosandbox.numpy.determine_type import is_casadi_type
from aerosandbox.numpy.typing import Vectorizable, Array


def _make_casadi_types_broadcastable(
    x1: Vectorizable, x2: Vectorizable
) -> tuple[Array, Array]:
    """Make two CasADi arrays broadcastable to a common shape.

    Parameters
    ----------
    x1 : Vectorizable
        First input array.
    x2 : Vectorizable
        Second input array.

    Returns
    -------
    tuple[Array, Array]
        Both arrays tiled to have the same (broadcast) shape.
    """

    def shape_2D(object: Vectorizable) -> tuple:
        shape = _onp.shape(object)
        if len(shape) == 0:
            return (1, 1)
        elif len(shape) == 1:
            return (1, shape[0])
        elif len(shape) == 2:
            return shape
        else:
            raise ValueError(
                "CasADi can't handle arrays with >2 dimensions, unfortunately."
            )

    x1_shape = shape_2D(x1)
    x2_shape = shape_2D(x2)
    shape = _onp.broadcast_shapes(x1_shape, x2_shape)

    x1_tiled = _cas.repmat(
        x1,
        shape[0] // x1_shape[0],
        shape[1] // x1_shape[1],
    )
    x2_tiled = _cas.repmat(
        x2,
        shape[0] // x2_shape[0],
        shape[1] // x2_shape[1],
    )

    return x1_tiled, x2_tiled


def add(x1: Vectorizable, x2: Vectorizable) -> Array:
    """Add arguments element-wise.

    Parameters
    ----------
    x1, x2 : Vectorizable
        The arrays to be added. Must be broadcastable to a common shape.

    Returns
    -------
    Array
        The element-wise sum of the inputs.

    See Also
    --------
    numpy.add : https://numpy.org/doc/stable/reference/generated/numpy.add.html
    """
    if not is_casadi_type(x1) and not is_casadi_type(x2):
        return _onp.add(x1, x2)
    else:
        x1, x2 = _make_casadi_types_broadcastable(x1, x2)
        return x1 + x2


def multiply(x1: Vectorizable, x2: Vectorizable) -> Array:
    """Multiply arguments element-wise.

    Parameters
    ----------
    x1, x2 : Vectorizable
        The arrays to be multiplied. Must be broadcastable to a common shape.

    Returns
    -------
    Array
        The element-wise product of the inputs.

    See Also
    --------
    numpy.multiply : https://numpy.org/doc/stable/reference/generated/numpy.multiply.html
    """
    if not is_casadi_type(x1) and not is_casadi_type(x2):
        return _onp.multiply(x1, x2)
    else:
        x1, x2 = _make_casadi_types_broadcastable(x1, x2)
        return x1 * x2


def mod(x1: Vectorizable, x2: Vectorizable) -> Array:
    """Return element-wise remainder of division.

    Parameters
    ----------
    x1 : Vectorizable
        Dividend array.
    x2 : Vectorizable
        Divisor array.

    Returns
    -------
    Array
        The element-wise remainder of the floor division of the inputs.
        Has the same sign as the divisor ``x2``.

    See Also
    --------
    numpy.mod : https://numpy.org/doc/stable/reference/generated/numpy.mod.html
    """
    if not is_casadi_type(x1) and not is_casadi_type(x2):
        return _onp.mod(x1, x2)

    else:
        out = _cas.fmod(x1, x2)
        # `fmod` returns a remainder with the sign of x1; NumPy's `mod` returns a
        # remainder with the sign of x2. Correct nonzero remainders of the wrong
        # sign (keying on the remainder itself, so that exact multiples map to 0).
        out = where(out * _cas.sign(x2) < 0, out + x2, out)
        return out


def centered_mod(x1: Vectorizable, x2: Vectorizable) -> Array:
    """Return element-wise remainder of division, centered on zero.

    Unlike ``mod``, this returns values in the range ``(-x2/2, x2/2]`` instead
    of ``[0, x2)``.

    Parameters
    ----------
    x1 : Vectorizable
        Dividend array.
    x2 : Vectorizable
        Divisor array.

    Returns
    -------
    Array
        The element-wise remainder, centered around zero.

    See Also
    --------
    mod : Standard modulo operation.
    """
    if not is_casadi_type(x1) and not is_casadi_type(x2):
        remainder = _onp.mod(x1, x2)
        return where(remainder > x2 / 2, remainder - x2, remainder)

    else:
        return _cas.remainder(x1, x2)
