"""Array creation and manipulation functions for the AeroSandbox NumPy-like interface.

This module provides array creation and manipulation functions that work with
both NumPy arrays and CasADi symbolic arrays, dispatching to the appropriate
backend at runtime based on input types.
"""

import numpy as _onp
import casadi as _cas
from typing import Any, Sequence
from aerosandbox.numpy.determine_type import is_casadi_type
from aerosandbox.numpy.typing import ArrayLike, Array, Scalar


def array(array_like: ArrayLike, dtype: type | None = None) -> Array:
    """Initialize a new array from array-like input.

    Create a NumPy array if possible; if the input contains CasADi types,
    create a CasADi array instead.

    Parameters
    ----------
    array_like : ArrayLike
        Input data (list, tuple, ndarray, or CasADi array).
    dtype : type, optional
        The desired data-type for the array. If provided and input contains
        CasADi types, this is ignored (CasADi determines its own dtype).

    Returns
    -------
    Array
        A NumPy array if input contains no CasADi types, otherwise a CasADi
        array.

    See Also
    --------
    numpy.array : https://numpy.org/doc/stable/reference/generated/numpy.array.html
    """
    if is_casadi_type(
        array_like, recursive=False
    ):  # If you were literally given a CasADi array, just return it
        # Handles inputs like cas.DM([1, 2, 3])
        return array_like

    elif not is_casadi_type(array_like, recursive=True) or dtype is not None:
        # If you were given a list of iterables that don't have CasADi types:
        # Handles inputs like [[1, 2, 3], [4, 5, 6]]
        return _onp.array(array_like, dtype=dtype)

    else:
        # Handles inputs like [[opti_var_1, opti_var_2], [opti_var_3, opti_var_4]]
        def make_row(contents: Sequence[Any]):
            try:
                return _cas.horzcat(*contents)
            except (TypeError, Exception):
                return contents

        return _cas.vertcat(*[make_row(row) for row in array_like])


def asarray(a: ArrayLike, dtype: type | None = None) -> Array:
    """Convert the input to an array.

    No copy is made if the input is already an ndarray with matching dtype.
    For CasADi arrays, this is a no-op (returns the input unchanged).

    Parameters
    ----------
    a : ArrayLike
        Input data, in any form that can be converted to an array. This
        includes lists, tuples, ndarrays, and CasADi arrays.
    dtype : data-type, optional
        By default, the data-type is inferred from the input data.

    Returns
    -------
    Array
        Array interpretation of ``a``. No copy is made if the input is
        already an ndarray with matching dtype, or if the input is a CasADi
        array.

    See Also
    --------
    numpy.asarray : https://numpy.org/doc/stable/reference/generated/numpy.asarray.html
    array : Always makes a copy for NumPy inputs.
    """
    if is_casadi_type(a, recursive=False):
        # CasADi array: no-op, return as-is
        return a
    elif not is_casadi_type(a, recursive=True):
        # Pure NumPy/Python: use numpy.asarray (no-copy if already ndarray)
        return _onp.asarray(a, dtype=dtype)
    else:
        # Mixed: contains CasADi types, must construct CasADi array
        return array(a, dtype=dtype)


def concatenate(arrays: Sequence[ArrayLike], axis: int = 0) -> Array:
    """Join a sequence of arrays along an existing axis.

    Return a NumPy array if all inputs are NumPy; otherwise return a CasADi
    array.

    Parameters
    ----------
    arrays : Sequence[ArrayLike]
        The arrays to concatenate. Must have the same shape except along the
        concatenation axis.
    axis : int, optional
        The axis along which the arrays will be joined. Default is 0.
        For CasADi arrays, only 0 or 1 are valid.

    Returns
    -------
    Array
        The concatenated array.

    Raises
    ------
    ValueError
        If CasADi arrays are used with an axis other than 0 or 1.

    See Also
    --------
    numpy.concatenate : https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html
    """
    if not is_casadi_type(arrays, recursive=True):
        return _onp.concatenate(arrays, axis=axis)

    else:
        if axis == 0:
            return _cas.vertcat(*arrays)
        elif axis == 1:
            return _cas.horzcat(*arrays)
        else:
            raise ValueError(
                "CasADi-backend arrays can only be 1D or 2D, so `axis` must be 0 or 1."
            )


def stack(arrays: Sequence[Array], axis: int = 0) -> Array:
    """Join a sequence of arrays along a new axis.

    Return a NumPy array if all inputs are NumPy; otherwise return a CasADi
    array.

    Parameters
    ----------
    arrays : Sequence[Array]
        Each array must have the same shape.
    axis : int, optional
        The axis along which the arrays will be stacked. Default is 0.
        For CasADi arrays, only 0, 1, -1, or -2 are valid.

    Returns
    -------
    Array
        The stacked array with one additional dimension.

    Raises
    ------
    ValueError
        If CasADi arrays are used with invalid axis, or if array shapes are
        incompatible for stacking.

    See Also
    --------
    numpy.stack : https://numpy.org/doc/stable/reference/generated/numpy.stack.html
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
                    raise ValueError(
                        "Can only stack 1D NumPy ndarrays alongside CasADi arrays!"
                    )

        if axis == 0 or axis == -2:
            return _cas.transpose(_cas.horzcat(*arrays))
        elif axis == 1 or axis == -1:
            return _cas.horzcat(*arrays)
        else:
            raise ValueError(
                "CasADi-backend arrays can only be 1D or 2D, so `axis` must be 0 or 1."
            )


def hstack(arrays: Sequence[ArrayLike]) -> _onp.ndarray:
    """Stack arrays in sequence horizontally (column wise).

    For NumPy arrays, behave like ``numpy.hstack``. CasADi arrays are not
    supported; use ``stack`` or ``concatenate`` instead.

    Parameters
    ----------
    arrays : Sequence[ArrayLike]
        The arrays to stack.

    Returns
    -------
    ndarray
        The horizontally stacked array.

    Raises
    ------
    ValueError
        If any input is a CasADi array.

    See Also
    --------
    numpy.hstack : https://numpy.org/doc/stable/reference/generated/numpy.hstack.html
    stack : Preferred alternative for mixed-backend arrays.
    concatenate : Preferred alternative for mixed-backend arrays.
    """
    if not is_casadi_type(arrays, recursive=True):
        return _onp.hstack(arrays)
    else:
        raise ValueError(
            "Use `np.stack()` or `np.concatenate()` instead of `np.hstack()` when dealing with mixed-backend arrays."
        )


def vstack(arrays: Sequence[ArrayLike]) -> _onp.ndarray:
    """Stack arrays in sequence vertically (row wise).

    For NumPy arrays, behave like ``numpy.vstack``. CasADi arrays are not
    supported; use ``stack`` or ``concatenate`` instead.

    Parameters
    ----------
    arrays : Sequence[ArrayLike]
        The arrays to stack.

    Returns
    -------
    ndarray
        The vertically stacked array.

    Raises
    ------
    ValueError
        If any input is a CasADi array.

    See Also
    --------
    numpy.vstack : https://numpy.org/doc/stable/reference/generated/numpy.vstack.html
    stack : Preferred alternative for mixed-backend arrays.
    concatenate : Preferred alternative for mixed-backend arrays.
    """
    if not is_casadi_type(arrays, recursive=True):
        return _onp.vstack(arrays)
    else:
        raise ValueError(
            "Use `np.stack()` or `np.concatenate()` instead of `np.vstack()` when dealing with mixed-backend arrays."
        )


def dstack(arrays: Sequence[ArrayLike]) -> _onp.ndarray:
    """Stack arrays in sequence depth wise (along third axis).

    For NumPy arrays, behave like ``numpy.dstack``. CasADi arrays are not
    supported; use ``stack`` or ``concatenate`` instead.

    Parameters
    ----------
    arrays : sequence of array_like
        The arrays to stack.

    Returns
    -------
    ndarray
        The depth-stacked array.

    Raises
    ------
    ValueError
        If any input is a CasADi array.

    See Also
    --------
    numpy.dstack : https://numpy.org/doc/stable/reference/generated/numpy.dstack.html
    stack : Preferred alternative for mixed-backend arrays.
    concatenate : Preferred alternative for mixed-backend arrays.
    """
    if not is_casadi_type(arrays, recursive=True):
        return _onp.dstack(arrays)
    else:
        raise ValueError(
            "Use `np.stack()` or `np.concatenate()` instead of `np.dstack()` when dealing with mixed-backend arrays."
        )


def length(array: ArrayLike | Scalar) -> int:
    """Return the length of a 1D-array-like object.

    An extension of ``len()`` that handles both NumPy arrays and CasADi arrays,
    as well as scalars (which return 1).

    Parameters
    ----------
    array : ArrayLike | Scalar
        Input array or scalar.

    Returns
    -------
    int
        The length of the array. Returns 1 for scalars. For CasADi arrays,
        returns the larger of the two dimensions (assuming column vectors).
    """
    if not is_casadi_type(array, recursive=False):
        try:
            return len(array)
        except TypeError:
            return 1

    else:
        array = asarray(array)  # Ensure array is Array for .shape access
        if array.shape[0] != 1:
            return array.shape[0]
        else:
            return array.shape[1]


def diag(v: ArrayLike, k: int = 0) -> Array:
    """Extract a diagonal or construct a diagonal array.

    If ``v`` is a 1D array, return a 2D array with ``v`` on the k-th diagonal.
    If ``v`` is a 2D array, return its k-th diagonal.

    Parameters
    ----------
    v : ArrayLike
        If 1D, elements to place on the diagonal. If 2D, array to extract
        diagonal from.
    k : int, optional
        Diagonal offset: k=0 is the main diagonal, k>0 is above the main
        diagonal, k<0 is below. Default is 0.

    Returns
    -------
    Array
        The extracted diagonal or constructed diagonal array.

    Raises
    ------
    NotImplementedError
        If extracting diagonal from a non-square CasADi matrix.

    See Also
    --------
    numpy.diag : https://numpy.org/doc/stable/reference/generated/numpy.diag.html
    """
    if not is_casadi_type(v, recursive=False):
        return _onp.diag(v, k=k)

    else:
        v = asarray(v)  # Ensure v is Array for .shape access
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
                return array([v[i, i + k] for i in range(n - k)])
            else:
                return array([v[i - k, i] for i in range(n + k)])

        else:
            raise NotImplementedError(
                "Haven't yet added logic for non-square matrices."
            )


def roll(
    a: ArrayLike,
    shift: int | tuple[int, ...],
    axis: int | tuple[int, ...] | None = None,
) -> Array:
    """Roll array elements along a given axis.

    Elements that roll beyond the last position are re-introduced at the first.

    Parameters
    ----------
    a : ArrayLike
        Input array.
    shift : int | tuple[int, ...]
        The number of places by which elements are shifted. If a tuple, then
        axis must be a tuple of the same size, and each of the given axes is
        shifted by the corresponding number. If an int while axis is a tuple
        of ints, then the same value is used for all given axes.
    axis : int | tuple[int, ...], optional
        Axis or axes along which elements are shifted. By default, the array
        is flattened before shifting, after which the original shape is
        restored.

    Returns
    -------
    Array
        Output array, with the same shape as ``a``.

    See Also
    --------
    numpy.roll : https://numpy.org/doc/stable/reference/generated/numpy.roll.html
    """
    if not is_casadi_type(a, recursive=False):
        return _onp.roll(a, shift, axis=axis)
    else:
        a = asarray(a)  # Ensure a is Array for .shape access
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


def max(a: ArrayLike, axis: int | None = None) -> Scalar | Array:
    """Return the maximum of an array or maximum along an axis.

    Parameters
    ----------
    a : ArrayLike
        Input array.
    axis : int, optional
        Axis along which to operate. By default, flattened input is used.
        For CasADi arrays, only None, 0, or 1 are valid.

    Returns
    -------
    Scalar | Array
        Maximum of ``a``. If axis is None, a scalar is returned. Otherwise,
        an array with the maxima along the specified axis.

    Raises
    ------
    ValueError
        If CasADi arrays are used with an invalid axis.

    See Also
    --------
    numpy.max : https://numpy.org/doc/stable/reference/generated/numpy.max.html
    """
    if not is_casadi_type(a, recursive=False):
        return _onp.max(
            a,
            axis=axis,
        )

    else:
        a = asarray(a)  # Ensure a is Array for .shape access
        if axis is None:
            return _cas.mmax(a)

        if axis == 0:
            if a.shape[1] == 1:
                return _cas.mmax(a)
            else:
                return array([_cas.mmax(a[:, i]) for i in range(a.shape[1])])

        elif axis == 1:
            if a.shape[0] == 1:
                return _cas.mmax(a)
            else:
                return array([_cas.mmax(a[i, :]) for i in range(a.shape[0])])

        else:
            raise ValueError(f"Invalid axis {axis} for CasADi array.")


def min(a: ArrayLike, axis: int | None = None) -> Scalar | Array:
    """Return the minimum of an array or minimum along an axis.

    Parameters
    ----------
    a : ArrayLike
        Input array.
    axis : int, optional
        Axis along which to operate. By default, flattened input is used.
        For CasADi arrays, only None, 0, or 1 are valid.

    Returns
    -------
    Scalar | Array
        Minimum of ``a``. If axis is None, a scalar is returned. Otherwise,
        an array with the minima along the specified axis.

    Raises
    ------
    ValueError
        If CasADi arrays are used with an invalid axis.

    See Also
    --------
    numpy.min : https://numpy.org/doc/stable/reference/generated/numpy.min.html
    """
    if not is_casadi_type(a, recursive=False):
        return _onp.min(
            a=a,
            axis=axis,
        )

    else:
        a = asarray(a)  # Ensure a is Array for .shape access
        if axis is None:
            return _cas.mmin(a)

        if axis == 0:
            if a.shape[1] == 1:
                return _cas.mmin(a)
            else:
                return array([_cas.mmin(a[:, i]) for i in range(a.shape[1])])

        elif axis == 1:
            if a.shape[0] == 1:
                return _cas.mmin(a)
            else:
                return array([_cas.mmin(a[i, :]) for i in range(a.shape[0])])

        else:
            raise ValueError(f"Invalid axis {axis} for CasADi array.")


def reshape(a: ArrayLike, newshape: int | tuple[int, ...], order: str = "C") -> Array:
    """Give a new shape to an array without changing its data.

    Parameters
    ----------
    a : ArrayLike
        Array to be reshaped.
    newshape : int | tuple[int, ...]
        The new shape should be compatible with the original shape. If an
        integer, the result will be a 1D array of that length. Use -1 to
        infer a dimension.
    order : {'C', 'F'}, optional
        Read and write elements using C-like (row-major) or Fortran-like
        (column-major) order. Default is 'C'.

    Returns
    -------
    Array
        Reshaped array.

    Raises
    ------
    ValueError
        If CasADi arrays are used with more than 2 dimensions.
    NotImplementedError
        If an order other than 'C' or 'F' is specified with CasADi arrays.

    See Also
    --------
    numpy.reshape : https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
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
            raise ValueError(
                "CasADi data types are limited to no more than 2 dimensions."
            )

        if order == "C":
            return _cas.reshape(a.T, newshape[::-1]).T
        elif order == "F":
            return _cas.reshape(a, newshape)
        else:
            raise NotImplementedError("Only C and F orders are supported.")


def ravel(a: ArrayLike, order: str = "C") -> Array:
    """Return a contiguous flattened array.

    Parameters
    ----------
    a : ArrayLike
        Input array.
    order : {'C', 'F'}, optional
        Read elements using C-like (row-major) or Fortran-like (column-major)
        order. Default is 'C'.

    Returns
    -------
    Array
        A 1D array containing the elements of ``a``.

    See Also
    --------
    numpy.ravel : https://numpy.org/doc/stable/reference/generated/numpy.ravel.html
    """
    if not is_casadi_type(a, recursive=False):
        return _onp.ravel(a, order=order)
    else:
        return reshape(a, -1, order=order)


def tile(A: ArrayLike, reps: tuple[int, ...]) -> Array:
    """Construct an array by repeating A the number of times given by reps.

    Parameters
    ----------
    A : ArrayLike
        The input array.
    reps : tuple[int, ...]
        The number of repetitions of A along each axis.

    Returns
    -------
    Array
        The tiled output array.

    Raises
    ------
    ValueError
        If CasADi arrays are used with more than 2 dimensions of repetition.

    See Also
    --------
    numpy.tile : https://numpy.org/doc/stable/reference/generated/numpy.tile.html
    """
    if not is_casadi_type(A, recursive=False):
        return _onp.tile(A, reps)
    else:
        if len(reps) == 1:
            return _cas.repmat(A, reps[0], 1)
        elif len(reps) == 2:
            return _cas.repmat(A, reps[0], reps[1])
        else:
            raise ValueError(
                "Cannot have >2D arrays when using CasADi numeric backend!"
            )


def zeros_like(
    a: ArrayLike,
    dtype: type | None = None,
    order: str = "K",
    subok: bool = True,
    shape: int | tuple[int, ...] | None = None,
) -> _onp.ndarray:
    """Return an array of zeros with the same shape and type as a given array.

    Parameters
    ----------
    a : ArrayLike
        The shape and data-type of ``a`` define these same attributes of the
        returned array.
    dtype : type, optional
        Overrides the data type of the result.
    order : {'C', 'F', 'A', 'K'}, optional
        Overrides the memory layout of the result. Default is 'K'.
    subok : bool, optional
        If True, use the sub-class type. Default is True.
    shape : int | tuple[int, ...], optional
        Overrides the shape of the result.

    Returns
    -------
    ndarray
        Array of zeros with the same shape and type as ``a``.

    See Also
    --------
    numpy.zeros_like : https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html
    """
    if not is_casadi_type(a, recursive=False):
        return _onp.zeros_like(a, dtype=dtype, order=order, subok=subok, shape=shape)
    else:
        return _onp.zeros(shape=length(a))


def ones_like(
    a: ArrayLike,
    dtype: type | None = None,
    order: str = "K",
    subok: bool = True,
    shape: int | tuple[int, ...] | None = None,
) -> _onp.ndarray:
    """Return an array of ones with the same shape and type as a given array.

    Parameters
    ----------
    a : ArrayLike
        The shape and data-type of ``a`` define these same attributes of the
        returned array.
    dtype : type, optional
        Overrides the data type of the result.
    order : {'C', 'F', 'A', 'K'}, optional
        Overrides the memory layout of the result. Default is 'K'.
    subok : bool, optional
        If True, use the sub-class type. Default is True.
    shape : int | tuple[int, ...], optional
        Overrides the shape of the result.

    Returns
    -------
    ndarray
        Array of ones with the same shape and type as ``a``.

    See Also
    --------
    numpy.ones_like : https://numpy.org/doc/stable/reference/generated/numpy.ones_like.html
    """
    if not is_casadi_type(a, recursive=False):
        return _onp.ones_like(a, dtype=dtype, order=order, subok=subok, shape=shape)
    else:
        return _onp.ones(shape=length(a))


def empty_like(
    prototype: ArrayLike,
    dtype: type | None = None,
    order: str = "K",
    subok: bool = True,
    shape: int | tuple[int, ...] | None = None,
) -> _onp.ndarray:
    """Return a new array with the same shape and type as a given array.

    Parameters
    ----------
    prototype : ArrayLike
        The shape and data-type of ``prototype`` define these same attributes
        of the returned array.
    dtype : type, optional
        Overrides the data type of the result.
    order : {'C', 'F', 'A', 'K'}, optional
        Overrides the memory layout of the result. Default is 'K'.
    subok : bool, optional
        If True, use the sub-class type. Default is True.
    shape : int | tuple[int, ...], optional
        Overrides the shape of the result.

    Returns
    -------
    ndarray
        Uninitialized array with the same shape and type as ``prototype``.

    See Also
    --------
    numpy.empty_like : https://numpy.org/doc/stable/reference/generated/numpy.empty_like.html
    """
    if not is_casadi_type(prototype, recursive=False):
        return _onp.empty_like(
            prototype, dtype=dtype, order=order, subok=subok, shape=shape
        )
    else:
        return zeros_like(prototype)


def full_like(
    a: ArrayLike,
    fill_value: Scalar,
    dtype: type | None = None,
    order: str = "K",
    subok: bool = True,
    shape: int | tuple[int, ...] | None = None,
) -> _onp.ndarray:
    """Return a full array with the same shape and type as a given array.

    Parameters
    ----------
    a : ArrayLike
        The shape and data-type of ``a`` define these same attributes of the
        returned array.
    fill_value : Scalar
        Fill value.
    dtype : type, optional
        Overrides the data type of the result.
    order : {'C', 'F', 'A', 'K'}, optional
        Overrides the memory layout of the result. Default is 'K'.
    subok : bool, optional
        If True, use the sub-class type. Default is True.
    shape : int | tuple[int, ...], optional
        Overrides the shape of the result.

    Returns
    -------
    ndarray
        Array of ``fill_value`` with the same shape and type as ``a``.

    See Also
    --------
    numpy.full_like : https://numpy.org/doc/stable/reference/generated/numpy.full_like.html
    """
    if not is_casadi_type(a, recursive=False):
        return _onp.full_like(
            a, fill_value, dtype=dtype, order=order, subok=subok, shape=shape
        )
    else:
        return fill_value * ones_like(a)


def assert_equal_shape(
    arrays: list[_onp.ndarray] | dict[str, _onp.ndarray],
) -> None:
    """Assert that all of the given arrays have the same shape.

    Parameters
    ----------
    arrays : list of ndarray, or dict of str to ndarray
        The arrays to be evaluated. Can be provided as:

        - A list, in which case a generic ValueError is thrown on mismatch.
        - A dictionary with name:array pairs, in which case the names are
          included in the error message.

    Raises
    ------
    ValueError
        If the arrays do not all have the same shape.
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
                raise ValueError(
                    f"The given arrays {namelist} do not have the same shape!"
                )
