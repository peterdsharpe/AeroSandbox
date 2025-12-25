"""Linear algebra functions for the AeroSandbox NumPy-like interface.

This module provides linear algebra functions that work with both NumPy
arrays and CasADi symbolic arrays.
"""
import numpy as _onp
import casadi as _cas
from aerosandbox.numpy.arithmetic_monadic import sum, abs
from aerosandbox.numpy.determine_type import is_casadi_type
from aerosandbox.numpy.array import asarray
from aerosandbox.numpy.typing import ArrayLike, Array, Scalar, VectorLike
from numpy.linalg import *


def inner(x: VectorLike, y: VectorLike, manual: bool = False) -> Scalar:
    """Compute the inner product of two arrays.

    Parameters
    ----------
    x : VectorLike
        First input array.
    y : VectorLike
        Second input array.
    manual : bool, optional
        If True, use a manual loop implementation. Default is False.

    Returns
    -------
    Scalar
        The inner product of ``x`` and ``y``.

    See Also
    --------
    numpy.inner : https://numpy.org/doc/stable/reference/generated/numpy.inner.html
    """
    if manual:
        return sum([xi * yi for xi, yi in zip(x, y)])

    if not is_casadi_type([x, y], recursive=True):
        return _onp.inner(x, y)

    else:
        return _cas.dot(x, y)


def outer(x: VectorLike, y: VectorLike, manual: bool = False) -> Array:
    """Compute the outer product of two vectors.

    Parameters
    ----------
    x : VectorLike
        First input vector.
    y : VectorLike
        Second input vector.
    manual : bool, optional
        If True, use a manual loop implementation. Default is False.

    Returns
    -------
    Array
        The outer product of ``x`` and ``y``, a 2D array of shape
        ``(len(x), len(y))``.

    See Also
    --------
    numpy.outer : https://numpy.org/doc/stable/reference/generated/numpy.outer.html
    """
    if manual:
        return [[xi * yi for yi in y] for xi in x]

    if not is_casadi_type([x, y], recursive=True):
        return _onp.outer(x, y)

    else:
        y = asarray(y)  # Ensure y is Array for .shape access
        if len(y.shape) == 1:  # Force y to be transposable if it's not.
            y = _onp.expand_dims(y, 1)
        return x @ y.T


def solve(A: ArrayLike, b: ArrayLike) -> Array:
    """Solve the linear system Ax=b for x.

    Parameters
    ----------
    A : ArrayLike, shape (M, M)
        Coefficient matrix.
    b : ArrayLike, shape (M,) or (M, N)
        Right-hand side vector or matrix.

    Returns
    -------
    Array, shape (M,) or (M, N)
        Solution to the system ``A @ x = b``.

    See Also
    --------
    numpy.linalg.solve : https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html
    """
    if not is_casadi_type([A, b]):
        return _onp.linalg.solve(A, b)

    else:
        return _cas.solve(A, b)


def inv(A: ArrayLike) -> Array:
    """Compute the inverse of a matrix.

    Parameters
    ----------
    A : ArrayLike, shape (M, M)
        Matrix to be inverted.

    Returns
    -------
    Array, shape (M, M)
        Inverse of the matrix ``A``.

    See Also
    --------
    numpy.linalg.inv : https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html
    """
    if not is_casadi_type(A):
        return _onp.linalg.inv(A)

    else:
        return _cas.inv(A)


def pinv(A: ArrayLike) -> Array:
    """Compute the Moore-Penrose pseudoinverse of a matrix.

    Parameters
    ----------
    A : ArrayLike, shape (M, N)
        Matrix to be pseudo-inverted.

    Returns
    -------
    Array, shape (N, M)
        The pseudo-inverse of the matrix ``A``.

    See Also
    --------
    numpy.linalg.pinv : https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html
    """
    if not is_casadi_type(A):
        return _onp.linalg.pinv(A)

    else:
        return _cas.pinv(A)


def det(A: ArrayLike) -> Scalar:
    """Compute the determinant of a matrix.

    Parameters
    ----------
    A : ArrayLike, shape (M, M)
        Input matrix.

    Returns
    -------
    Scalar
        Determinant of ``A``.

    See Also
    --------
    numpy.linalg.det : https://numpy.org/doc/stable/reference/generated/numpy.linalg.det.html
    """
    if not is_casadi_type(A):
        return _onp.linalg.det(A)

    else:
        return _cas.det(A)


def norm(
    x: ArrayLike,
    ord: int | float | str | None = None,
    axis: int | tuple[int, int] | None = None,
    keepdims: bool = False,
) -> Scalar | Array:
    """Compute matrix or vector norm.

    Parameters
    ----------
    x : ArrayLike
        Input array.
    ord : int | float | str, optional
        Order of the norm. Default is None (2-norm for vectors, Frobenius
        for matrices).
    axis : int | tuple[int, int], optional
        Axis along which to compute the norm. For CasADi arrays, only -1, 0,
        or 1 are valid.
    keepdims : bool, optional
        If True, the reduced axis is retained as a dimension of size one.
        Default is False.

    Returns
    -------
    Scalar | Array
        Norm of the array.

    Raises
    ------
    ValueError
        If CasADi arrays are used with an unsupported axis or ord.

    See Also
    --------
    numpy.linalg.norm : https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
    """
    if not is_casadi_type(x):
        return _onp.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)

    else:
        x = asarray(x)  # Ensure x is Array for .shape access
        # Figure out which axis, if any, to take a vector norm about.
        if axis is not None:
            if not (axis == 0 or axis == 1 or axis == -1):
                raise ValueError("`axis` must be -1, 0, or 1 for CasADi types.")
        elif x.shape[0] == 1:
            axis = 1
        elif x.shape[1] == 1:
            axis = 0

        if ord is None:
            if axis is not None:
                ord = 2
            else:
                ord = "fro"

        if ord == 1:
            # norm = _cas.norm_1(x)
            norm = sum(abs(x), axis=axis)
        elif ord == 2:
            # norm = _cas.norm_2(x)
            norm = sum(x**2, axis=axis) ** 0.5
        elif ord == "fro" or ord == "frobenius":
            norm = _cas.norm_fro(x)
        elif ord == "inf" or _onp.isinf(ord):
            norm = _cas.norm_inf(x)
        else:
            try:
                norm = sum(abs(x) ** ord, axis=axis) ** (1 / ord)
            except Exception as e:
                print(e)
                raise ValueError(
                    "Couldn't interpret `ord` sensibly! Tried to interpret it as a floating-point order "
                    "as a last-ditch effort, but that didn't work."
                )

        if keepdims:
            new_shape = list(x.shape)
            new_shape[axis] = 1
            return _cas.reshape(norm, new_shape)
        else:
            return norm


def inv_symmetric_3x3(
    m11: Scalar,
    m22: Scalar,
    m33: Scalar,
    m12: Scalar,
    m23: Scalar,
    m13: Scalar,
) -> tuple[Scalar, Scalar, Scalar, Scalar, Scalar, Scalar]:
    """Explicitly compute the inverse of a symmetric 3x3 matrix.

    Input matrix (note symmetry)::

        [m11, m12, m13]
        [m12, m22, m23]
        [m13, m23, m33]

    Output matrix (note symmetry)::

        [a11, a12, a13]
        [a12, a22, a23]
        [a13, a23, a33]

    Parameters
    ----------
    m11, m22, m33 : Scalar
        Diagonal elements of the symmetric matrix.
    m12, m23, m13 : Scalar
        Off-diagonal elements (m12=m21, m23=m32, m13=m31).

    Returns
    -------
    tuple[Scalar, Scalar, Scalar, Scalar, Scalar, Scalar]
        The 6 unique elements of the inverse matrix, in the same order as
        the inputs.

    References
    ----------
    .. [1] https://math.stackexchange.com/questions/233378/
    """
    det = (
        m11 * (m33 * m22 - m23**2)
        - m12 * (m33 * m12 - m23 * m13)
        + m13 * (m23 * m12 - m22 * m13)
    )
    inv_det = 1 / det
    a11 = m33 * m22 - m23**2
    a12 = m13 * m23 - m33 * m12
    a13 = m12 * m23 - m13 * m22

    a22 = m33 * m11 - m13**2
    a23 = m12 * m13 - m11 * m23

    a33 = m11 * m22 - m12**2

    a11 = a11 * inv_det
    a12 = a12 * inv_det
    a13 = a13 * inv_det
    a22 = a22 * inv_det
    a23 = a23 * inv_det
    a33 = a33 * inv_det

    return a11, a22, a33, a12, a23, a13
