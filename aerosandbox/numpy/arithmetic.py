import numpy as onp
import casadi as cas
from aerosandbox.numpy.array import length
from .determine_type import is_casadi_type


def sum(x):
    """Returns the sum of a vector x."""
    if is_casadi_type(x):
        return cas.sum1(x)
    else:
        return onp.sum(x)


def mean(x):
    """Returns the mean of a vector x."""
    return sum(x) / length(x)

def cumsum(x, **kwargs):
    """Return the cumulative sum of the elements."""
    
    try:
        return onp.cumsum(x, **kwargs)
    except Exception:
        return cas.cumsum(x, **kwargs)