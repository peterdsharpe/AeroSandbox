from typing import Union
import casadi as _cas
import numpy as _onp
from aerosandbox.numpy.array import length, roll, concatenate
from aerosandbox.numpy.calculus import diff


def integrate_discrete_segments(
        f: Union[_onp.ndarray, _cas.MX],
        x: Union[_onp.ndarray, _cas.MX] = None,
        multiply_by_dx: bool = True,
        method: str = "trapezoidal",
        method_endpoints: str = "lower_order",
):
    """
    Given a set of points (x_i, f_i) from a function, computes the integral of that function over each set of
    adjacent points.

    In general, N points will yield N-1 integrals (one for each "gap" between points).

    Args:
        f: A 1D array of function values.
        x: A 1D array of x-values. If not specified, defaults to the indices of f.
        method: The integration method to use. Options are:
            - "forward_euler"
            - "backward_euler"
            - "trapezoidal" (default)
            - "forward_simpson"
            - "backward_simpson"
            - "cubic"
            Note that some methods, like "cubic", approximate each segment interval by looking beyond just the integral itself (i.e., f(a) and f(b)),
             and so are not possible near the endpoints of the array.
        method_endpoints: The integration method to use at the endpoints. Options are:
            - "lower_order" (default)
            - "ignore" (i.e. return the integral of the interior points only - note that this may result in a different number of integrals than segments!)
            - "periodic"

    """
    x_is_specified = (x is not None)
    if not x_is_specified:
        x = _onp.arange(length(f))

    dx = x[1:] - x[:-1]

    if method in ["forward_euler", "forward", "left", "left_riemann"]:
        avg_f = f[:-1]

        degree = 0  # Refers to the highest degree of the polynomial that the method is exact for.
        remaining_endpoint_intervals = (0, 0)

    elif method in ["backward_euler", "backward", "right", "right_riemann"]:
        avg_f = f[1:]

        degree = 0
        remaining_endpoint_intervals = (0, 0)

    elif method in ["trapezoidal", "trapezoid", "trapz", "midpoint"]:
        avg_f = (f[1:] + f[:-1]) / 2

        degree = 1
        remaining_endpoint_intervals = (0, 0)

    elif method in ["forward_simpson", "forward_simpsons"]:
        x1 = x[:-2]
        x2 = x[1:-1]
        x3 = x[2:]

        f1 = f[:-2]
        f2 = f[1:-1]
        f3 = f[2:]

        h = x2 - x1
        hp = x3 - x2

        # q1 = 0  # Integration lower bound
        # q2 = 1  # Integration upper bound
        q3 = 1 + hp / h

        avg_f = (
                        f1
                        - f3
                        + 3 * q3 ** 2 * (f1 + f2)
                        - 2 * q3 * (2 * f1 + f2)
                ) / (
                        6 * q3 * (q3 - 1)
                )

        degree = 2
        remaining_endpoint_intervals = (0, 1)

    elif method in ["backward_simpson", "backward_simpsons"]:
        x1 = x[:-2]
        x2 = x[1:-1]
        x3 = x[2:]

        f1 = f[:-2]
        f2 = f[1:-1]
        f3 = f[2:]

        h = x3 - x2
        hm = x2 - x1

        q1 = -hm / h
        # q2 = 0  # Integration lower bound
        # q3 = 1  # Integration upper bound

        avg_f = (
                        -f1
                        + f2
                        + 3 * q1 ** 2 * (f2 + f3)
                        - 2 * q1 * (2 * f2 + f3)
                ) / (
                        6 * q1 * (q1 - 1)
                )

        degree = 2
        remaining_endpoint_intervals = (1, 0)

    elif method in ["cubic", "cubic_spline"]:
        x1 = x[:-3]
        x2 = x[1:-2]
        x3 = x[2:-1]
        x4 = x[3:]

        f1 = f[:-3]
        f2 = f[1:-2]
        f3 = f[2:-1]
        f4 = f[3:]

        h = x3 - x2
        hm = x2 - x1
        hp = x4 - x3

        q1 = -hm / h
        # q2 = 0  # Integration lower bound
        # q3 = 1  # Integration upper bound
        q4 = 1 + hp / h

        avg_f = (
                        6 * q1 ** 3 * q4 ** 2 * (f2 + f3)
                        - 4 * q1 ** 3 * q4 * (2 * f2 + f3)
                        + 2 * q1 ** 3 * (f2 - f4)
                        - 6 * q1 ** 2 * q4 ** 3 * (f2 + f3)
                        + 3 * q1 ** 2 * q4 * (3 * f2 + f3)
                        + 3 * q1 ** 2 * (f4 - f2)
                        + 4 * q1 * q4 ** 3 * (2 * f2 + f3)
                        - 3 * q1 * q4 ** 2 * (3 * f2 + f3)
                        + q1 * (f2 - f4)
                        + 2 * q4 ** 3 * (f1 - f2)
                        + 3 * q4 ** 2 * (-f1 + f2)
                        + q4 * (f1 - f2)
                ) / (
                        12 * q1 * q4 * (q1 - 1) * (q1 - q4) * (q4 - 1)
                )

        degree = 3
        remaining_endpoint_intervals = (1, 1)

    else:
        raise ValueError(f"Invalid method '{method}'.")

    if method_endpoints == "lower_order":
        if degree >= 3:
            method_endpoints = "simpson"
        else:
            method_endpoints = "trapezoidal"

        if method_endpoints == "simpson":

            # Do the left endpoint(s)
            avg_f_left_intervals = integrate_discrete_segments(
                f=f[:2 + remaining_endpoint_intervals[0]],
                x=x[:2 + remaining_endpoint_intervals[0]],
                multiply_by_dx=False,
                method="forward_simpson",
                method_endpoints="ignore",
            )
            avg_f_right_intervals = integrate_discrete_segments(
                f=f[-(2 + remaining_endpoint_intervals[1]):],
                x=x[-(2 + remaining_endpoint_intervals[1]):],
                multiply_by_dx=False,
                method="backward_simpson",
                method_endpoints="ignore",
            )

            avg_f = concatenate((
                avg_f_left_intervals,
                avg_f,
                avg_f_right_intervals,
            ))

        elif method_endpoints == "trapezoidal":

            # Do the left endpoint(s)
            avg_f_left_intervals = integrate_discrete_segments(
                f=f[:1 + remaining_endpoint_intervals[0]],
                x=x[:1 + remaining_endpoint_intervals[0]],
                multiply_by_dx=False,
                method="trapezoidal",
                method_endpoints="ignore",
            )
            avg_f_right_intervals = integrate_discrete_segments(
                f=f[-(1 + remaining_endpoint_intervals[1]):],
                x=x[-(1 + remaining_endpoint_intervals[1]):],
                multiply_by_dx=False,
                method="trapezoidal",
                method_endpoints="ignore",
            )

            avg_f = concatenate((
                avg_f_left_intervals,
                avg_f,
                avg_f_right_intervals,
            ))

    elif method_endpoints == "ignore":
        pass

    elif method_endpoints == "periodic":
        raise NotImplementedError("Periodic integration is not yet implemented.")

    else:
        raise ValueError(f"Invalid method_endpoints '{method_endpoints}'.")

    if multiply_by_dx:
        if x_is_specified:
            return avg_f * dx
        else:
            return avg_f
    else:
        return avg_f


if __name__ == '__main__':
    import aerosandbox as asb
    import aerosandbox.numpy as np

    np.random.seed(0)

    degree = 4
    coeffs = np.random.randn(degree + 1)
    a = 1
    b = 3


    def f(x):
        out = 0
        for i in range(degree + 1):
            out += coeffs[i] * x ** i

        return out


    from scipy import integrate

    exact = integrate.quad(
        f,
        a,
        b,
        epsrel=1e-15,
    )[0]
    print(f"exact: {exact}")

    x = np.linspace(a, b, 100)
    f = f(x)

    approx_intervals = integrate_discrete_segments(
        f=f,
        x=x,
        multiply_by_dx=True,
        # method="trapz",
        method="cubic",
        # method_endpoints="ignore",
    )

    integral = np.sum(approx_intervals)

    print(f"approx: {integral}")
    print(f"error: {integral - exact}")
