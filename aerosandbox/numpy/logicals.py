import numpy as _onp
import casadi as _cas
from aerosandbox.numpy.determine_type import is_casadi_type


def clip(
        x,
        min,
        max
):
    """
    Clip a value to a range.
    Args:
        x: Value to clip.
        min: Minimum value to clip to.
        max: Maximum value to clip to.

    Returns:

    """
    return _onp.fmin(_onp.fmax(x, min), max)


def logical_and(x1, x2):
    """
    Compute the truth value of x1 AND x2 element-wise.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.logical_and.html
    """
    if not is_casadi_type([x1, x2], recursive=True):
        return _onp.logical_and(x1, x2)

    else:
        return _cas.logic_and(x1, x2)


def logical_or(x1, x2):
    """
    Compute the truth value of x1 OR x2 element-wise.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.logical_or.html
    """
    if not is_casadi_type([x1, x2], recursive=True):
        return _onp.logical_or(x1, x2)

    else:
        return _cas.logic_or(x1, x2)


def logical_not(x):
    """
    Compute the truth value of NOT x element-wise.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.logical_not.html
    """
    if not is_casadi_type(x, recursive=False):
        return _onp.logical_not(x)

    else:
        return _cas.logic_not(x)


def all(a):  # TODO add axis functionality
    """
    Test whether all array elements along a given axis evaluate to True.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.all.html
    """
    if not is_casadi_type(a, recursive=False):
        return _onp.all(a)

    else:
        return _cas.logic_all(a)


def any(a):  # TODO add axis functionality
    """
    Test whether any array element along a given axis evaluates to True.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.any.html
    """
    if not is_casadi_type(a, recursive=False):
        return _onp.any(a)

    else:
        return _cas.logic_any(a)
