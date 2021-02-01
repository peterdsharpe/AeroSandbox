import numpy as np
import casadi as cas


def sum(x):
    """Returns the sum of a vector x."""
    try:
        return np.sum(x)
    except Exception:
        return cas.sum1(x)
