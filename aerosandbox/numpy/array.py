import numpy as _onp
import casadi as _cas
from typing import List, Tuple, Dict, Union, Sequence
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


def concatenate(arrays: Sequence, axis: int = 0):
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


def stack(arrays: Sequence, axis: int = 0):
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
    if not is_casadi_type(v, recursive=False):
        return _onp.diag(v, k=k)

    else:

        if 1 in v.shape:  # If v is a 1D array, construct a diagonal matrix
            if v.shape[0] == 1:
                v = v.T

            if k == 0:
                return _cas.diag(v)

            else:
                n = v.shape[0]
                res = type(v).zeros(n + abs(k), n + abs(k))
                for i in range(n):
                    if k >= 0:
                        res[i, i + k] = v[i]
                    else:
                        res[i - k, i] = v[i]
                return res

        elif v.shape[0] == v.shape[1]:  # If v is a square matrix, extract the diagonal

            n = v.shape[0]

            if k >= 0:
                return array([
                    v[i, i + k]
                    for i in range(n - k)
                ])
            else:
                return array([
                    v[i - k, i]
                    for i in range(n + k)
                ])

        else:
            raise NotImplementedError("Haven't yet added logic for non-square matrices.")


def roll(a, shift, axis: int = None):
    """
    Roll array elements along a given axis.

    Elements that roll beyond the last position are re-introduced at the first.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.roll.html

    Parameters
    ----------
    a : array_like

        Input array.

    shift : int or tuple of ints

        The number of places by which elements are shifted. If a tuple, then axis must be a tuple of the same size,
        and each of the given axes is shifted by the corresponding number. If an int while axis is a tuple of ints,
        then the same value is used for all given axes.

    axis : int or tuple of ints, optional

        Axis or axes along which elements are shifted. By default, the array is flattened before shifting,
        after which the original shape is restored.

    Returns
    -------
    res : ndarray
        Output array, with the same shape as a.

    """
    if not is_casadi_type(a, recursive=False):
        return _onp.roll(a, shift, axis=axis)
    else:
        if axis is None:
            a_flat = reshape(a, -1)
            result = roll(a_flat, shift, axis=0)
            return reshape(result, a.shape)
        elif isinstance(axis, int):
            shift = shift % a.shape[axis]  # shift can be negative
            if shift != 0:
                slice1 = [slice(None)] * 2
                slice1[axis] = slice(-shift, None)
                slice2 = [slice(None)] * 2
                slice2[axis] = slice(-shift)
                result = concatenate([a[tuple(slice1)], a[tuple(slice2)]], axis=axis)
            else:
                result = a
            return result
        elif isinstance(axis, tuple):
            result = a
            if not isinstance(shift, tuple):
                shift = (shift,) * len(axis)
            for ax, sh in zip(axis, shift):
                result = roll(result, sh, ax)
            return result
        else:
            raise ValueError("'axis' must be None, an integer or a tuple of integers")


def max(a, axis=None):
    """
    Return the maximum of an array or maximum along an axis.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.max.html
    """

    if not is_casadi_type(a, recursive=False):
        return _onp.max(
            a,
            axis=axis,
        )

    else:

        if axis is None:
            return _cas.mmax(a)

        if axis == 0:
            if a.shape[1] == 1:
                return _cas.mmax(a)
            else:
                return array([
                    _cas.mmax(a[:, i])
                    for i in range(a.shape[1])
                ])

        elif axis == 1:
            if a.shape[0] == 1:
                return _cas.mmax(a)
            else:
                return array([
                    _cas.mmax(a[i, :])
                    for i in range(a.shape[0])
                ])

        else:
            raise ValueError(f'Invalid axis {axis} for CasADi array.')


def min(a, axis=None):
    """
    Return the minimum of an array or minimum along an axis.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.min.html
    """

    if not is_casadi_type(a, recursive=False):
        return _onp.min(
            a=a,
            axis=axis,
        )

    else:

        if axis is None:
            return _cas.mmin(a)

        if axis == 0:
            if a.shape[1] == 1:
                return _cas.mmin(a)
            else:
                return array([
                    _cas.mmin(a[:, i])
                    for i in range(a.shape[1])
                ])

        elif axis == 1:
            if a.shape[0] == 1:
                return _cas.mmin(a)
            else:
                return array([
                    _cas.mmin(a[i, :])
                    for i in range(a.shape[0])
                ])

        else:
            raise ValueError(f'Invalid axis {axis} for CasADi array.')


def reshape(a, newshape, order='C'):
    """
    Gives a new shape to an array without changing its data.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
    """

    if not is_casadi_type(a, recursive=False):
        return _onp.reshape(a, newshape, order=order)
    else:
        if isinstance(newshape, int):
            newshape = (newshape, 1)

        elif len(newshape) == 1:
            newshape = (newshape[0], 1)

        elif len(newshape) == 2:
            newshape = tuple(newshape)

        elif len(newshape) > 2:
            raise ValueError("CasADi data types are limited to no more than 2 dimensions.")

        if order == "C":
            return _cas.reshape(a.T, newshape[::-1]).T
        elif order == "F":
            return _cas.reshape(a, newshape)
        else:
            raise NotImplementedError("Only C and F orders are supported.")


def ravel(a, order='C'):
    """
    Returns a contiguous flattened array.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.ravel.html
    """
    if not is_casadi_type(a, recursive=False):
        return _onp.ravel(a, order=order)
    else:
        return reshape(a, -1, order=order)


def tile(A, reps):
    """
    Construct an array by repeating A the number of times given by reps.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.tile.html
    """
    if not is_casadi_type(A, recursive=False):
        return _onp.tile(A, reps)
    else:
        if len(reps) == 1:
            return _cas.repmat(A, reps[0], 1)
        elif len(reps) == 2:
            return _cas.repmat(A, reps[0], reps[1])
        else:
            raise ValueError("Cannot have >2D arrays when using CasADi numeric backend!")


def zeros_like(a, dtype=None, order='K', subok=True, shape=None):
    """Return an array of zeros with the same shape and type as a given array."""
    if not is_casadi_type(a, recursive=False):
        return _onp.zeros_like(a, dtype=dtype, order=order, subok=subok, shape=shape)
    else:
        return _onp.zeros(shape=length(a))


def ones_like(a, dtype=None, order='K', subok=True, shape=None):
    """Return an array of ones with the same shape and type as a given array."""
    if not is_casadi_type(a, recursive=False):
        return _onp.ones_like(a, dtype=dtype, order=order, subok=subok, shape=shape)
    else:
        return _onp.ones(shape=length(a))


def empty_like(prototype, dtype=None, order='K', subok=True, shape=None):
    """Return a new array with the same shape and type as a given array."""
    if not is_casadi_type(prototype, recursive=False):
        return _onp.empty_like(prototype, dtype=dtype, order=order, subok=subok, shape=shape)
    else:
        return zeros_like(prototype)


def full_like(a, fill_value, dtype=None, order='K', subok=True, shape=None):
    """Return a full array with the same shape and type as a given array."""
    if not is_casadi_type(a, recursive=False):
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
