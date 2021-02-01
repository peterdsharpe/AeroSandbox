import numpy as np
import casadi as cas


def inner(x, y):
    """Return the inner product of vectors x and y."""
    try:
        return np.inner(x, y)
    except Exception:
        return x.T @ y


def outer(x, y):
    """Return the outer product of vectors x and y."""
    try:
        return np.outer(x, y)
    except Exception:
        return x @ y.T


def linear_solve(A, b):  # TODO get this working
    """
    Solve the linear system Ax=b for x.
    Args:
        A: A square matrix.
        b: A vector representing the RHS of the linear system.

    Returns: The solution vector x.

    """
    try:
        return np.linalg.solve(A, b)
    except Exception:
        return cas.solve(A, b)


def norm(x):
    """
    Returns the L2-norm of a vector x.
    """
    try:
        return np.linalg.norm(x)
    except Exception:
        return cas.norm_2(x)
