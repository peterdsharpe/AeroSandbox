import numpy as _onp
import casadi as _cas
from typing import List, Tuple, Dict, Union
from aerosandbox.numpy.determine_type import is_casadi_type


def array(array_like, dtype=None):
    """
    Initializes a new array. Creates a NumPy array if possible; if not, creates a CasADi array.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.array.html
    """
    if is_casadi_type(array_like, recursive=False):  # If you were literally given a CasADi array, just return it
        # Handles inputs like cas.DM([1, 2, 3])
        return array_like

    elif not is_casadi_type(array_like,
                            recursive=True) or dtype is not None:
        # If you were given a list of iterables that don't have CasADi types:
        # Handles inputs like [[1, 2, 3], [4, 5, 6]]
        return _onp.array(array_like, dtype=dtype)

    else:
        # Handles inputs like [[opti_var_1, opti_var_2], [opti_var_3, opti_var_4]]
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
    Returns the length of an 1D-array-like object. An extension of len() with slightly different functionality.
    Args:
        array:

    Returns:

    """
    if not is_casadi_type(array, recursive=False):
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

        if 1 in v.shape:
            return _cas.diag(v)
        elif v.shape[0] == v.shape[1]:
            raise NotImplementedError("Should be super possible, just haven't had the need yet.")
        else:
            raise ValueError("Cannot return the diagonal of a non-square matrix.")


def roll(a, shift, axis: int = None):
    """
    Roll array elements along a given axis.

    Elements that roll beyond the last position are re-introduced at the first.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.roll.html

    Parameters
    ----------
    a : array_like
        Input array.
    shift : int
        The number of places by which elements are shifted.

    Returns
    -------
    res : ndarray
        Output array, with the same shape as a.

    """
    if not is_casadi_type(a):
        return _onp.roll(a, shift, axis=axis)
    else:  # TODO add some checking to make sure shift < len(a), or shift is modulo'd down by len(a).
        # assert shift < a.shape[axis]
        if 1 in a.shape and axis == 0:
            return _cas.vertcat(a[-shift, :], a[:-shift, :])
        elif axis == 0:
            return _cas.vertcat(a.T[:, -shift], a.T[:, :-shift]).T
        elif axis == 1:
            return _cas.horzcat(a[:, -shift], a[:, :-shift])
        elif axis is None:
            return roll(a, shift=shift, axis=0)
        else:
            raise Exception("CasADi types can only be up to 2D, so `axis` must be None, 0, or 1.")


def max(a):
    """
    Returns the maximum value of an array.
    """

    try:
        return _onp.max(a)
    except TypeError:
        return _cas.mmax(a)


def min(a):
    """
    Returns the minimum value of an array.
    """

    try:
        return _onp.min(a)
    except TypeError:
        return _cas.mmin(a)


def reshape(a, newshape):
    """Gives a new shape to an array without changing its data."""

    if not is_casadi_type(a):
        return _onp.reshape(a, newshape)
    else:
        if isinstance(newshape, int):
            newshape = (newshape, 1)

        if len(newshape) == 1:
            newshape = (newshape[0], 1)

        if len(newshape) > 2:
            raise ValueError("CasADi data types are limited to no more than 2 dimensions.")

        return _cas.reshape(a.T, newshape[::-1]).T


def zeros_like(a, dtype=None, order='K', subok=True, shape=None):
    """Return an array of zeros with the same shape and type as a given array."""
    if not is_casadi_type(a):
        return _onp.zeros_like(a, dtype=dtype, order=order, subok=subok, shape=shape)
    else:
        return _onp.zeros(shape=length(a))


def ones_like(a, dtype=None, order='K', subok=True, shape=None):
    """Return an array of ones with the same shape and type as a given array."""
    if not is_casadi_type(a):
        return _onp.ones_like(a, dtype=dtype, order=order, subok=subok, shape=shape)
    else:
        return _onp.ones(shape=length(a))


def empty_like(prototype, dtype=None, order='K', subok=True, shape=None):
    """Return a new array with the same shape and type as a given array."""
    if not is_casadi_type(prototype):
        return _onp.empty_like(prototype, dtype=dtype, order=order, subok=subok, shape=shape)
    else:
        return zeros_like(prototype)


def full_like(a, fill_value, dtype=None, order='K', subok=True, shape=None):
    """Return a full array with the same shape and type as a given array."""
    if not is_casadi_type(a):
        return _onp.full_like(a, fill_value, dtype=dtype, order=order, subok=subok, shape=shape)
    else:
        return fill_value * ones_like(a)


def assert_equal_shape(
        arrays: Union[List[_onp.ndarray], Dict[str, _onp.ndarray]],
) -> None:
    """
    Assert that all of the given arrays are the same shape. If this is not true, raise a ValueError.

    Args: arrays: The arrays to be evaluated.

            Can be provided as a:

                * List, in which case a generic ValueError is thrown

                * Dictionary consisting of name:array pairs for key:value, in which case the names are given in the ValueError.

    Returns: None. Throws an error if leng

    """
    try:
        names = arrays.keys()
        arrays = list(arrays.values())
    except AttributeError:
        names = None

    def get_shape(array):
        try:
            return array.shape
        except AttributeError:  # If it's a float/int
            return ()

    shape = get_shape(arrays[0])

    for array in arrays[1:]:
        if not get_shape(array) == shape:
            if names is None:
                raise ValueError("The given arrays do not have the same shape!")
            else:
                namelist = ", ".join(names)
                raise ValueError(f"The given arrays {namelist} do not have the same shape!")
