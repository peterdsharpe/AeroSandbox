import numpy as _onp
import casadi as _cas
from typing import Tuple, Iterable, Union
from aerosandbox.numpy.conditionals import where

from aerosandbox.numpy.determine_type import is_casadi_type


def _make_casadi_types_broadcastable(x1, x2):
    def shape_2D(object: Union[float, int, Iterable, _onp.ndarray]) -> Tuple:
        shape = _onp.shape(object)
        if len(shape) == 0:
            return (1, 1)
        elif len(shape) == 1:
            return (1, shape[0])
        elif len(shape) == 2:
            return shape
        else:
            raise ValueError("CasADi can't handle arrays with >2 dimensions, unfortunately.")

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


def add(
        x1, x2
):
    if not is_casadi_type(x1) and not is_casadi_type(x2):
        return _onp.add(x1, x2)
    else:
        x1, x2 = _make_casadi_types_broadcastable(x1, x2)
        return x1 + x2


def multiply(
        x1, x2
):
    if not is_casadi_type(x1) and not is_casadi_type(x2):
        return _onp.multiply(x1, x2)
    else:
        x1, x2 = _make_casadi_types_broadcastable(x1, x2)
        return x1 * x2


def mod(x1, x2):
    """
    Return element-wise remainder of division.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.mod.html
    """
    if not is_casadi_type(x1) and not is_casadi_type(x2):
        return _onp.mod(x1, x2)

    else:
        out = _cas.fmod(x1, x2)
        out = where(
            x1 < 0,
            out + x2,
            out
        )
        return out


def centered_mod(x1, x2):
    """
    Return element-wise remainder of division, centered on zero.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.mod.html
    """
    if not is_casadi_type(x1) and not is_casadi_type(x2):
        remainder = _onp.mod(x1, x2)
        return where(
            remainder > x2 / 2,
            remainder - x2,
            remainder
        )

    else:
        return _cas.remainder(x1, x2)
