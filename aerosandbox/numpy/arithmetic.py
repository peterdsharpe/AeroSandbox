import numpy as _onp
import casadi as _cas
from aerosandbox.numpy.determine_type import is_casadi_type


def sum(x, axis: int = None):
    """
    Sum of array elements over a given axis.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.sum.html
    """
    if not is_casadi_type(x):
        return _onp.sum(x, axis=axis)

    else:
        if axis == 0:
            return _cas.sum1(x).T

        elif axis == 1:
            return _cas.sum2(x)
        elif axis is None:
            return sum(sum(x, axis=0), axis=0)
        else:
            raise ValueError("CasADi types can only be up to 2D, so `axis` must be None, 0, or 1.")


def mean(x, axis: int = None):
    """
    Compute the arithmetic mean along the specified axis.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.mean.html
    """
    if not is_casadi_type(x):
        return _onp.mean(x, axis=axis)

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
    if not is_casadi_type(x):
        return _onp.abs(x)

    else:
        return _cas.fabs(x)


# TODO trace()

# def cumsum(x, axis: int = None):
#     """
#     Return the cumulative sum of the elements along a given axis.
#
#     See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html
#     """
#
#     if not is_casadi_type(x):
#         return _onp.cumsum(x, axis=axis)
#
#     else:
#         raise NotImplementedError
#         if axis is None:
#             return _cas.cumsum(_onp.flatten(x))
