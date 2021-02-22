import numpy as _onp
import casadi as _cas
from typing import List, Tuple
from aerosandbox.numpy.determine_type import is_casadi_type


def array(array_like, dtype=None):
    """
    Initializes a new array. Creates a NumPy array if possible; if not, creates a CasADi array.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.array.html
    """
    if not is_casadi_type(array_like, recursive=True):
        return _onp.array(array_like, dtype=dtype)

    else:
        def make_row(contents: List):
            try:
                return _cas.horzcat(*contents)
            except (TypeError, Exception):
                return contents

        return _cas.vertcat(
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
        return _onp.concatenate(arrays, axis=axis)

    else:
        if axis == 0:
            return _cas.vertcat(*arrays)
        elif axis == 1:
            return _cas.horzcat(*arrays)
        else:
            raise ValueError("CasADi-backend arrays can only be 1D or 2D, so `axis` must be 0 or 1.")


def stack(arrays: Tuple, axis: int = 0):
    """
    Join a sequence of arrays along a new axis. Returns a NumPy array if possible; if not, returns a CasADi array.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.stack.html
    """
    if not is_casadi_type(arrays, recursive=True):
        return _onp.stack(arrays, axis=axis)

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
            return _cas.transpose(_cas.horzcat(*arrays))
        elif axis == 1 or axis == -1:
            return _cas.horzcat(*arrays)
        else:
            raise ValueError("CasADi-backend arrays can only be 1D or 2D, so `axis` must be 0 or 1.")


def hstack(arrays):
    if not is_casadi_type(arrays, recursive=True):
        return _onp.hstack(arrays)
    else:
        raise ValueError(
            "Use `np.stack()` or `np.concatenate()` instead of `np.hstack()` when dealing with mixed-backend arrays.")


def vstack(arrays):
    if not is_casadi_type(arrays, recursive=True):
        return _onp.vstack(arrays)
    else:
        raise ValueError(
            "Use `np.stack()` or `np.concatenate()` instead of `np.vstack()` when dealing with mixed-backend arrays.")


def dstack(arrays):
    if not is_casadi_type(arrays, recursive=True):
        return _onp.dstack(arrays)
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
    if not is_casadi_type(array):
        try:
            return len(array)
        except TypeError:
            return 1

    else:
        if array.shape[0] != 1:
            return array.shape[0]
        else:
            return array.shape[1]

def diag(v, k=0):
    """
    Extract a diagonal or construct a diagonal array.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.diag.html
    """
    if not is_casadi_type(v):
        return _onp.diag(v, k=k)

    else:
        if k != 0:
            raise NotImplementedError("Should be super possible, just haven't had the need yet.")

        if v.shape[0] == 1 or v.shape[1] == 1:
            return _cas.diag(v)
        elif v.shape[0] == v.shape[1]:
            raise NotImplementedError("Should be super possible, just haven't had the need yet.")
        else:
            raise ValueError("Cannot return the diagonal of a non-square matrix.")