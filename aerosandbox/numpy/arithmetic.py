import numpy as onp
import casadi as cas
from aerosandbox.numpy.array import length
from .determine_type import is_casadi_type


def sum(x, axis: int = None):
    """
    Sum of array elements over a given axis.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.sum.html
    """
    if not is_casadi_type(x):
        return onp.sum(x, axis=axis)
    else:
        if axis == 0:
            return cas.sum1(x)
        elif axis == 1:
            return cas.sum2(x)
        elif axis is None:
            return sum(sum(x, axis=0), axis=1)
        else:
            raise ValueError("CasADi types can only be up to 2D, so `axis` must be None, 0, or 1.")


def mean(x, axis: int = None):
    """
    Compute the arithmetic mean along the specified axis.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.mean.html
    """
    if not is_casadi_type(x):
        return onp.mean(x, axis=axis)
    else:
        if axis == 0:
            return sum(x, axis=0) / x.shape[0]
        elif axis == 1:
            return sum(x, axis=1) / x.shape[1]
        elif axis is None:
            return mean(mean(x, axis=0), axis=1)
        else:
            raise ValueError("CasADi types can only be up to 2D, so `axis` must be None, 0, or 1.")


def abs(x):
    try:
        return onp.abs(x)
    except TypeError:
        return onp.fabs(x)
