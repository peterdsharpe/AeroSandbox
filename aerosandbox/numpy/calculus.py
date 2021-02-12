def diff(x):
    """
    Computes the approximate local derivative of `x` via finite differencing assuming unit spacing.
    Can be viewed as the opposite of trapz().

    Args:
        x: The vector-like object (1D np.ndarray, cas.MX) to be integrated.

    Returns: A vector of length N-1 with each piece corresponding to the difference between one value and the next.

    """
    return x[1:] - x[:-1]


def trapz(x, modify_endpoints=False):
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
