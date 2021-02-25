import numpy as _onp
import casadi as _cas
from aerosandbox.numpy.determine_type import is_casadi_type


def diff(a, n=1, axis=-1):
    """
    Calculate the n-th discrete difference along the given axis.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.diff.html
    """
    if not is_casadi_type(a):
        return _onp.diff(a, n=n, axis=axis)

    else:
        if axis != -1:
            raise NotImplementedError("This could be implemented, but haven't had the need yet.")

        result = a
        for i in range(n):
            result = _cas.diff(a)
        return result


def trapz(x, modify_endpoints=False): # TODO unify with NumPy trapz, this is different
    """
    Computes each piece of the approximate integral of `x` via the trapezoidal method with unit spacing.
    Can be viewed as the opposite of diff().

    Args:
        x: The vector-like object (1D np.ndarray, cas.MX) to be integrated.

    Returns: A vector of length N-1 with each piece corresponding to the mean value of the function on the interval
        starting at index i.

    """
    integral = (
                       x[1:] + x[:-1]
               ) / 2
    if modify_endpoints:
        integral[0] = integral[0] + x[0] * 0.5
        integral[-1] = integral[-1] + x[-1] * 0.5

    return integral
