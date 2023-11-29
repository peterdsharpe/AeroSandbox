import numpy as _onp
import casadi as _cas
from aerosandbox.numpy.arithmetic_monadic import sum, abs
from aerosandbox.numpy.determine_type import is_casadi_type
from numpy.linalg import *


def inner(x, y, manual=False):
    """
    Inner product of two arrays.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.inner.html
    """
    if manual:
        return sum([xi * yi for xi, yi in zip(x, y)])

    if not is_casadi_type([x, y], recursive=True):
        return _onp.inner(x, y)

    else:
        return _cas.dot(x, y)


def outer(x, y, manual=False):
    """
    Compute the outer product of two vectors.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.outer.html
    """
    if manual:
        return [
            [
                xi * yi
                for yi in y
            ]
            for xi in x
        ]

    if not is_casadi_type([x, y], recursive=True):
        return _onp.outer(x, y)

    else:
        if len(y.shape) == 1:  # Force y to be transposable if it's not.
            y = _onp.expand_dims(y, 1)
        return x @ y.T


def solve(A, b):  # TODO get this working
    """
    Solve the linear system Ax=b for x.
    Args:
        A: A square matrix.
        b: A vector representing the RHS of the linear system.

    Returns: The solution vector x.

    """
    if not is_casadi_type([A, b]):
        return _onp.linalg.solve(A, b)

    else:
        return _cas.solve(A, b)


def inv(A):
    """
    Returns the inverse of the matrix A.

    See: https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html
    """
    if not is_casadi_type(A):
        return _onp.linalg.inv(A)

    else:
        return _cas.inv(A)


def pinv(A):
    """
    Returns the Moore-Penrose pseudoinverse of the matrix A.

    See: https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html
    """
    if not is_casadi_type(A):
        return _onp.linalg.pinv(A)

    else:
        return _cas.pinv(A)


def det(A):
    """
    Returns the determinant of the matrix A.

    See: https://numpy.org/doc/stable/reference/generated/numpy.linalg.det.html
    """
    if not is_casadi_type(A):
        return _onp.linalg.det(A)

    else:
        return _cas.det(A)


def norm(x, ord=None, axis=None, keepdims=False):
    """
    Matrix or vector norm.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
    """
    if not is_casadi_type(x):
        return _onp.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)

    else:

        # Figure out which axis, if any, to take a vector norm about.
        if axis is not None:
            if not (
                    axis == 0 or
                    axis == 1 or
                    axis == -1
            ):
                raise ValueError("`axis` must be -1, 0, or 1 for CasADi types.")
        elif x.shape[0] == 1:
            axis = 1
        elif x.shape[1] == 1:
            axis = 0

        if ord is None:
            if axis is not None:
                ord = 2
            else:
                ord = 'fro'

        if ord == 1:
            # norm = _cas.norm_1(x)
            norm = sum(
                abs(x),
                axis=axis
            )
        elif ord == 2:
            # norm = _cas.norm_2(x)
            norm = sum(
                x ** 2,
                axis=axis
            ) ** 0.5
        elif ord == 'fro' or ord == "frobenius":
            norm = _cas.norm_fro(x)
        elif ord == 'inf' or _onp.isinf(ord):
            norm = _cas.norm_inf()
        else:
            try:
                norm = sum(
                    abs(x) ** ord,
                    axis=axis
                ) ** (1 / ord)
            except Exception as e:
                print(e)
                raise ValueError("Couldn't interpret `ord` sensibly! Tried to interpret it as a floating-point order "
                                 "as a last-ditch effort, but that didn't work.")

        if keepdims:
            new_shape = list(x.shape)
            new_shape[axis] = 1
            return _cas.reshape(norm, new_shape)
        else:
            return norm

def inv_symmetric_3x3(
        m11,
        m22,
        m33,
        m12,
        m23,
        m13,
):
    """
    Explicitly computes the inverse of a symmetric 3x3 matrix.

    Input matrix (note symmetry):

    [m11, m12, m13]
    [m12, m22, m23]
    [m13, m23, m33]

    Output matrix (note symmetry):

    [a11, a12, a13]
    [a12, a22, a23]
    [a13, a23, a33]

    From https://math.stackexchange.com/questions/233378/inverse-of-a-3-x-3-covariance-matrix-or-any-positive-definite-pd-matrix
    """
    det = (
            m11 * (m33 * m22 - m23 ** 2) -
            m12 * (m33 * m12 - m23 * m13) +
            m13 * (m23 * m12 - m22 * m13)
    )
    inv_det = 1 / det
    a11 = m33 * m22 - m23 ** 2
    a12 = m13 * m23 - m33 * m12
    a13 = m12 * m23 - m13 * m22

    a22 = m33 * m11 - m13 ** 2
    a23 = m12 * m13 - m11 * m23

    a33 = m11 * m22 - m12 ** 2

    a11 = a11 * inv_det
    a12 = a12 * inv_det
    a13 = a13 * inv_det
    a22 = a22 * inv_det
    a23 = a23 * inv_det
    a33 = a33 * inv_det

    return a11, a22, a33, a12, a23, a13
