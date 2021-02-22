import numpy as _onp
import casadi as _cas
from aerosandbox.numpy.determine_type import is_casadi_type

def inner(x, y):
    """Return the inner product of vectors x and y."""
    if not is_casadi_type([x, y], recursive=True):
        return _onp.inner(x, y)

    else:
        if len(x.shape) == 1:  # Force x to be transposable if it's not.
            x = _onp.expand_dims(x, 1)
        return x.T @ y


def outer(x, y):
    """Return the outer product of vectors x and y."""
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


def norm(x):
    """
    Returns the L2-norm of a vector x.
    """
    if not is_casadi_type(x):
        return _onp.linalg.norm(x)

    else:
        return _cas.norm_2(x)
