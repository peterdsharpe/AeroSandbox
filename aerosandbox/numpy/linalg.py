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
        y = _cas.DM(y) # Force y to be transposable if it's not.
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
        is_vector = (
                x.shape[0] == 1 or
                x.shape[1] == 1
        )

        if ord is None:
            if is_vector:
                ord = 2
            else:
                ord = 'fro'

        if ord == 1:
            return _cas.norm_1(x)
        elif ord == 2:
            return _cas.norm_2(x)
        elif ord == 'fro':
            return _cas.norm_fro(x)
        elif ord == _onp.Inf:
            return _cas.norm_inf()
        else:
            try:
                return sum(abs(x) ** ord) ** (1 / ord)
            except Exception as e:
                print(e)
                raise ValueError("Couldn't interpret `ord` sensibly! Tried to interpret it as a floating-point order "
                                 "as a last-ditch effort, but that didn't work.")
