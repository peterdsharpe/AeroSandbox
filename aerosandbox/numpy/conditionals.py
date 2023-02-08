import numpy as _onp
import casadi as _cas
from aerosandbox.numpy.determine_type import is_casadi_type


def where(
        condition,
        value_if_true,
        value_if_false,
):
    """
    Return elements chosen from x or y depending on condition.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.where.html
    """
    if not is_casadi_type([condition, value_if_true, value_if_false], recursive=True):
        return _onp.where(
            condition,
            value_if_true,
            value_if_false
        )
    else:
        return _cas.if_else(
            condition,
            value_if_true,
            value_if_false
        )


def maximum(
        x1,
        x2,
):
    """
    Element-wise maximum of two arrays.

    Note: not differentiable at the crossover point, will cause issues if you try to optimize across a crossover.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.maximum.html
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
        x1,
        x2,
):
    """
    Element-wise minimum of two arrays.

    Note: not differentiable at the crossover point, will cause issues if you try to optimize across a crossover.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.minimum.html
    """
    if not is_casadi_type([x1, x2], recursive=True):
        return _onp.minimum(x1, x2)
    else:
        return where(
            x1 <= x2,
            x1,
            x2,
        )
