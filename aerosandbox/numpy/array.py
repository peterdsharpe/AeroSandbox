import numpy as onp
import casadi as cas
from typing import List, Tuple
from aerosandbox.numpy.determine_type import is_casadi_type


def array(array_like, dtype=None):
    """
    Initializes a new array. Creates a NumPy array if possible; if not, creates a CasADi array.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.array.html
    """
    if not is_casadi_type(array_like, recursive=True):
        return onp.array(array_like, dtype=dtype)

    else:
        def make_row(contents: List):
            try:
                return cas.horzcat(*contents)
            except (TypeError, Exception):
                return contents

        return cas.vertcat(
            *[
                make_row(row)
                for row in array_like
            ]
        )


def concatenate(arrays: Tuple, axis: int = 0):
    """
    Join a sequence of arrays along an existing axis. Returns a NumPy array if possible; if not, returns a CasADi array.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html
    """

    if not is_casadi_type(arrays, recursive=True):
        return onp.concatenate(arrays, axis=axis)

    else:
        if axis == 0:
            return cas.vertcat(*arrays)
        elif axis == 1:
            return cas.horzcat(*arrays)
        else:
            raise ValueError("CasADi-backend arrays can only be 1D or 2D, so `axis` must be 0 or 1.")


def stack(arrays: Tuple, axis: int = 0):
    """
    Join a sequence of arrays along a new axis. Returns a NumPy array if possible; if not, returns a CasADi array.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.stack.html
    """
    if not is_casadi_type(arrays, recursive=True):
        return onp.stack(arrays, axis=axis)

    else:
        ### Validate stackability
        for array in arrays:
            if is_casadi_type(array, recursive=False):
                if not array.shape[1] == 1:
                    raise ValueError("Can only stack Nx1 CasADi arrays!")
            else:
                if not len(array.shape) == 1:
                    raise ValueError("Can only stack 1D NumPy ndarrays alongside CasADi arrays!")

        if axis == 0 or axis == -2:
            return cas.transpose(cas.horzcat(*arrays))
        elif axis == 1 or axis == -1:
            return cas.horzcat(*arrays)
        else:
            raise ValueError("CasADi-backend arrays can only be 1D or 2D, so `axis` must be 0 or 1.")


def hstack(arrays):
    if not is_casadi_type(arrays, recursive=True):
        return onp.hstack(arrays)
    else:
        raise ValueError(
            "Use `np.stack()` or `np.concatenate()` instead of `np.hstack()` when dealing with mixed-backend arrays.")


def vstack(arrays):
    if not is_casadi_type(arrays, recursive=True):
        return onp.vstack(arrays)
    else:
        raise ValueError(
            "Use `np.stack()` or `np.concatenate()` instead of `np.vstack()` when dealing with mixed-backend arrays.")


def dstack(arrays):
    if not is_casadi_type(arrays, recursive=True):
        return onp.dstack(arrays)
    else:
        raise ValueError(
            "Use `np.stack()` or `np.concatenate()` instead of `np.dstack()` when dealing with mixed-backend arrays.")


def length(array) -> int:
    """
    Returns the length of an 1D-array-like object.
    Args:
        array:

    Returns:

    """
    try:
        return len(array)
    except TypeError:  # array has no function len() -> either float, int, or CasADi type
        try:
            if len(array.shape) >= 1:
                return array.shape[0]
            else:
                raise AttributeError
        except AttributeError:  # array has no attribute shape -> either float or int
            return 1


if __name__ == '__main__':
    array
