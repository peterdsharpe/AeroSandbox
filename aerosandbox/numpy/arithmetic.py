import numpy as np
import casadi as cas
from aerosandbox.numpy.array import length


def sum(x):
    """Returns the sum of a vector x."""
    try:
        return np.sum(x)
    except Exception:
        return cas.sum1(x)


def mean(x):
    """Returns the mean of a vector x."""
    return sum(x) / length(x)
