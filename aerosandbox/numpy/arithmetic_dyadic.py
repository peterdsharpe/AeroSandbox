import numpy as _onp
import casadi as _cas
from aerosandbox.numpy.determine_type import is_casadi_type

def add(
        x1, x2
):
    if not is_casadi_type(x1) and not is_casadi_type(x2):
        return _onp.add(x1, x2)
    else:
        shape = _onp.broadcast_shapes(x1.shape, x2.shape)
        x1_tiled = _cas.repmat(
            x1,
            shape[0] // x1.shape[0],
            shape[1] // x1.shape[1]
        )
        x2_tiled = _cas.repmat(
            x2,
            shape[0] // x2.shape[0],
            shape[1] // x2.shape[1]
        )
        return x1_tiled + x2_tiled

def multiply(
        x1, x2
):
    if not is_casadi_type(x1) and not is_casadi_type(x2):
        return _onp.multiply(x1, x2)
    else:
        shape = _onp.broadcast_shapes(x1.shape, x2.shape)
        x1_tiled = _cas.repmat(
            x1,
            shape[0] // x1.shape[0],
            shape[1] // x1.shape[1]
        )
        x2_tiled = _cas.repmat(
            x2,
            shape[0] // x2.shape[0],
            shape[1] // x2.shape[1]
        )
        return x1_tiled * x2_tiled


def mod(x1, x2):
    """
    Return element-wise remainder of division.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.mod.html
    """
    if not is_casadi_type(x1) and not is_casadi_type(x2):
        return _onp.mod(x1, x2)

    else:
        return _cas.mod(x1, x2)
