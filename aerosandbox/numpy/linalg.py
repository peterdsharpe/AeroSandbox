import numpy as _onp
import casadi as _cas
from aerosandbox.numpy.arithmetic import sum, abs
from aerosandbox.numpy.determine_type import is_casadi_type


def inner(x, y):
    """
    Inner product of two arrays.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.inner.html
    """
    if not is_casadi_type([x, y], recursive=True):
        return _onp.inner(x, y)

    else:
        return _cas.dot(x, y)


def outer(x, y):
    """
    Compute the outer product of two vectors.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.outer.html
    """
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


def norm(x, ord=None, axis=None):
    """
    Matrix or vector norm.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
    """
    if not is_casadi_type(x):
        return _onp.linalg.norm(x, ord=ord, axis=axis)

    else:

        # Figure out which axis, if any, to take a vector norm about.
        if axis is not None:
            if not (
                axis==0 or
                axis==1 or
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
            # return _cas.norm_1(x)
            return sum(
                abs(x),
                axis=axis
            )
        elif ord == 2:
            # return _cas.norm_2(x)
            return sum(
                x ** 2,
                axis=axis
            ) ** 0.5
        elif ord == 'fro':
            return _cas.norm_fro(x)
        elif np.isinf(ord):
            return _cas.norm_inf()
        else:
            try:
                return sum(
                    abs(x) ** ord,
                    axis=axis
                ) ** (1 / ord)
            except Exception as e:
                print(e)
                raise ValueError("Couldn't interpret `ord` sensibly! Tried to interpret it as a floating-point order "
                                 "as a last-ditch effort, but that didn't work.")
