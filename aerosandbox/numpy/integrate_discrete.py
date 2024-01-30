from typing import Union
import casadi as _cas
import numpy as _onp
from aerosandbox.numpy.array import length, roll, concatenate
from aerosandbox.numpy.calculus import diff


def integrate_discrete_intervals(
        f: Union[_onp.ndarray, _cas.MX],
        x: Union[_onp.ndarray, _cas.MX] = None,
        multiply_by_dx: bool = True,
        method: str = "trapezoidal",
        method_endpoints: str = "lower_order",
):
    """
    Given a set of sampled points (x_i, f_i) from a function, computes the integral of that function over each set of
    adjacent points ("intervals"). Does this via a reconstruction approach, with several methods available.

    In general, N points will yield N-1 integrals (one for each "interval" between points).

    Args:
        f: A 1D array of function values.

        x: A 1D array of x-values where the function was evaluated. If not specified, defaults to the indices of f.
            Should be the same length as f and should be monotonically increasing (i.e. x[i] < x[i+1], with no duplicated points).

        multiply_by_dx: Whether to multiply the integral by the width of the segment. Defaults to True.
            - If True, summing the integrals will yield the integral of the function over the entire domain (x[0] to x[-1])
            - If False, you can think of the output as the "average function value" over each interval.

        method: The integration method to use. Options are:
            - "forward_euler"
            - "backward_euler"
            - "trapezoidal" (default)
            - "forward_simpson"
            - "backward_simpson"
            - "cubic"
            Note that some methods, like "cubic", approximate each segment interval by looking beyond just the integral itself (i.e., f(a) and f(b)),
             and so are not possible near the endpoints of the array.

        method_endpoints: The integration method to use at the endpoints, for those higher-order methods that require handling. Options are:
            - "lower_order" (default)
            - "ignore" (i.e. return the integral of the interior points only - note that this may result in a different number of integrals than segments!)
            - "periodic"

    """
    # Determine if an x-array was specified, and calculate dx.
    x_is_specified = (x is not None)
    if not x_is_specified:
        x = _onp.arange(length(f))

    dx = x[1:] - x[:-1]

    method = str(method).lower().replace(" ", "_")

    # Implement integration methods
    if method in ["forward_euler", "forward", "euler_forward", "left", "left_riemann"]:
        avg_f = f[:-1]

        degree = 0  # Refers to the highest degree of the polynomial that the method is exact for.
        remaining_endpoint_intervals = (0, 0)

    elif method in ["backward_euler", "backward", "euler_backward", "right", "right_riemann"]:
        avg_f = f[1:]

        degree = 0
        remaining_endpoint_intervals = (0, 0)

    elif method in ["trapezoidal", "trapezoid", "trapz", "midpoint"]:
        if method == "midpoint":
            raise PendingDeprecationWarning(
                "The 'midpoint' method will be deprecated at a future point, since 'trapezoidal' is the more accurate term here.")

        avg_f = (f[1:] + f[:-1]) / 2

        degree = 1
        remaining_endpoint_intervals = (0, 0)

    elif method in ["forward_simpson", "simpson_forward", "simpson"]:
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
                        f1 - f3
                        + 3 * q3 ** 2 * (f1 + f2)
                        - 2 * q3 * (2 * f1 + f2)
                ) / (
                        6 * q3 * (q3 - 1)
                )

        degree = 2
        remaining_endpoint_intervals = (0, 1)

    elif method in ["backward_simpson", "simpson_backward"]:
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
                        f2 - f1
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
                        + 3 * q4 ** 2 * (f2 - f1)
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

            if remaining_endpoint_intervals[0] != 0:
                avg_f_left_intervals = integrate_discrete_intervals(
                    f=f[:2 + remaining_endpoint_intervals[0]],
                    x=x[:2 + remaining_endpoint_intervals[0]],
                    multiply_by_dx=False,
                    method="forward_simpson",
                    method_endpoints="ignore",
                )
                avg_f = concatenate((
                    avg_f_left_intervals,
                    avg_f
                ))

            if remaining_endpoint_intervals[1] != 0:
                avg_f_right_intervals = integrate_discrete_intervals(
                    f=f[-(2 + remaining_endpoint_intervals[1]):],
                    x=x[-(2 + remaining_endpoint_intervals[1]):],
                    multiply_by_dx=False,
                    method="backward_simpson",
                    method_endpoints="ignore",
                )
                avg_f = concatenate((
                    avg_f,
                    avg_f_right_intervals,
                ))

        elif method_endpoints == "trapezoidal":

            if remaining_endpoint_intervals[0] != 0:
                avg_f_left_intervals = integrate_discrete_intervals(
                    f=f[:1 + remaining_endpoint_intervals[0]],
                    x=x[:1 + remaining_endpoint_intervals[0]],
                    multiply_by_dx=False,
                    method="trapezoidal",
                    method_endpoints="ignore",
                )

                avg_f = concatenate((
                    avg_f_left_intervals,
                    avg_f
                ))

            if remaining_endpoint_intervals[1] != 0:
                avg_f_right_intervals = integrate_discrete_intervals(
                    f=f[-(1 + remaining_endpoint_intervals[1]):],
                    x=x[-(1 + remaining_endpoint_intervals[1]):],
                    multiply_by_dx=False,
                    method="trapezoidal",
                    method_endpoints="ignore",
                )

                avg_f = concatenate((
                    avg_f,
                    avg_f_right_intervals,
                ))

        else:
            raise ValueError(f"Invalid method_endpoints '{method_endpoints}'.")

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


def integrate_discrete_squared_curvature(
        f: Union[_onp.ndarray, _cas.MX],
        x: Union[_onp.ndarray, _cas.MX] = None,
        method: str = "hybrid_simpson_cubic",
):
    """
    Given a set of sampled points (x_i, f_i) from a function f(x), computes the following quantity:

        int_{x[0]}^{x[-1]} (f''(x))^2 dx

    This is useful for regularization of smooth curves (i.e., encouraging smooth functions as optimization results).

    Performs this through one of several reconstruction-based methods, specified by `method`:

        * "cubic": On each interval, reconstructs a piecewise cubic polynomial. This cubic is the unique polynomial
        that passes through the two points at the endpoints of the interval, plus the next point beyond each endpoint
        of the interval (i.e., 4 points in total). Numerically, this cubic is obtained using Bernstein polynomial
        reconstruction, so it is numerically stable. This cubic is then analytically differentiated twice, squared,
        and integrated over the interval. At the ends of the overall array, where this "look beyond" strategy is not
        possible, a one-sided cubic is used instead (i.e., looks beyond the interval at one end only and uses two
        extra points from this side).

        * "simpson": On each interval, makes two unique quadratic reconstructions:

            * One reconstruction that uses the two points at the endpoints of the interval, plus the next point beyond
            the right endpoint of the interval (i.e., 3 points in total).

            * One reconstruction that uses the two points at the endpoints of the interval, plus the next point beyond
            the left endpoint of the interval (i.e., 3 points in total).

            These two quadratics are then analytically differentiated twice, squared, and integrated over the
            interval. This requires much less calculation, since the quadratics have uniform curvature over the
            interval, causing a lot of things to simplify. The result is then computed by combining the results of this
            process for the two quadratic reconstructions.

            This is similar to a Simpson's rule integration, balanced between the two sides of the interval. In
            frequency-domain testing, this method appears to be more accurate than the "cubic" strategy at every
            frequency, with less computational effort. Thus, it should be preferred to the "cubic" strategy.

        * "hybrid_simpson_cubic": First, starts out by estimating the first derivative of the function at each point
        in the array (including endpoints) using a quadratic reconstruction. (See `numpy.gradient()` for more
        information or source code on this; this code uses `numpy.gradient()` directly for this step.) Then,
        reconstructs a cubic polynomial on each interval, with the following boundary conditions:

            * The cubic passes through the two points at the endpoints of the interval.

            * The cubic has the same first derivative as the precomputed derivatives at the endpoints of the interval.

            This cubic is then analytically differentiated twice, squared, and integrated over the interval.

            In frequency-domain testing, this method is also more accurate than the "cubic" strategy at every
            frequency. Compared to the "simpson" strategy, it is more accurate at high frequencies and less accurate
            at low frequencies. Because the goal of this function is to be used as a regularization term,
            which should be more sensitive to high-frequency oscillations, this method is preferred to the "simpson"
            strategy. This method is also preferred as its estimate tends to err high rather than low, which serves
            well as a regularization strategy. (It is still convergent to the true value in the high-sample-rate limit.)

    """
    # Determine if an x-array was specified, and calculate dx.
    x_is_specified = (x is not None)
    if not x_is_specified:
        x = _onp.arange(length(f))

    if method in ["cubic", "cubic_spline"]:
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

        dfm = f2 - f1
        df = f3 - f2
        dfp = f4 - f3

        ### The following section computes the integral of the squared second derivative of the cubic spline interpolant
        ### for the "middle" intervals (i.e. not the first or last intervals).
        ### Code is generated by sympy; here s_i variables represent common subexpressions.
        s0 = hm ** 2
        s1 = hp ** 2
        s2 = h + hm
        s3 = h ** 2
        s4 = hp ** 6
        s5 = h ** 6
        s6 = hp ** 5
        s7 = h ** 3
        s8 = 3 * s7
        s9 = hp ** 4
        s10 = h ** 4
        s11 = 4 * s10
        s12 = hp ** 3
        s13 = 3 * h ** 5
        s14 = hm ** 6
        s15 = hm ** 5
        s16 = hm ** 4
        s17 = hm ** 3
        s18 = hm * s10 * s12
        s19 = hp * s10 * s17
        s20 = s12 * s17
        s21 = s0 * s12 * s8
        s22 = s1 * s17 * s8
        s23 = 3 * s0 * s1 * s10 + s21 + s22
        s24 = 2 * h
        s25 = 6 * s3
        s26 = 7 * s7
        s27 = 3 * s3
        s28 = -s20 * s27
        s29 = 3 * h
        middle_intervals = 4 * (df ** 2 * (
                s0 * s27 * s9 + s0 * s29 * s6 + s0 * s4 + s1 * s14 + s1 * s15 * s29 + s1 * s16 * s27 - s12 * s16 * s29 - 2 * s16 * s9 - s17 * s29 * s9 + s23 + s28) + df * dfm * (
                                        2 * h * s17 * s9 - hm * s1 * s13 - hm * s24 * s4 - hm * s25 * s6 - hm * s26 * s9 + 3 * s0 * s3 * s9 + s1 * s17 * s7 - 6 * s18 + s21 - s28) + df * dfp * (
                                        2 * h * s12 * s16 - hp * s0 * s13 - hp * s14 * s24 - hp * s15 * s25 - hp * s16 * s26 + s0 * s12 * s7 + 3 * s1 * s16 * s3 - 6 * s19 + s22 - s28) + dfm ** 2 * (
                                        s1 * s5 + s11 * s9 + s12 * s13 + s3 * s4 + s6 * s8) + dfm * dfp * (
                                        hm * hp * s5 - s18 - s19 - 2 * s20 * s3 - s23) + dfp ** 2 * (
                                        s0 * s5 + s11 * s16 + s13 * s17 + s14 * s3 + s15 * s8)) / (
                                   h * s0 * s1 * s2 ** 2 * (h + hp) ** 2 * (hp + s2) ** 2)

        ### Now we compute the integral for the first interval.
        h_f = h[slice(0, 1)]
        hm_f = hm[slice(0, 1)]
        hp_f = hp[slice(0, 1)]
        df_f = df[slice(0, 1)]
        dfm_f = dfm[slice(0, 1)]
        dfp_f = dfp[slice(0, 1)]

        s0 = h_f ** 2
        s1 = hp_f ** 2
        s2 = h_f + hm_f
        s3 = hp_f ** 6
        s4 = df_f * dfm_f
        s5 = hm_f ** 6
        s6 = 2 * dfp_f
        s7 = df_f * s6
        s8 = h_f ** 6
        s9 = dfm_f * dfp_f
        s10 = 4 * s9
        s11 = df_f ** 2
        s12 = hm_f ** 2
        s13 = dfm_f ** 2
        s14 = dfp_f ** 2
        s15 = hm_f ** 3
        s16 = hp_f ** 5
        s17 = 3 * s11
        s18 = hp_f ** 4
        s19 = hm_f ** 4
        s20 = 4 * s19
        s21 = hm_f ** 5
        s22 = hp_f ** 3
        s23 = h_f ** 3
        s24 = 6 * s13
        s25 = h_f ** 4
        s26 = h_f ** 5
        s27 = 3 * s14
        s28 = df_f * dfp_f
        s29 = -dfm_f
        s30 = df_f * h_f
        s31 = 15 * df_f
        s32 = 2 * df_f
        s33 = dfm_f * hm_f
        s34 = 3 * dfm_f
        s35 = -s34
        s36 = dfp_f * hp_f
        s37 = 3 * s12
        s38 = 5 * s11
        s39 = 3 * s0
        s40 = -3 * s28 + s9
        s41 = 18 * s11
        s42 = -s7
        first_interval = 4 * (
                -2 * h_f * hm_f * s3 * s4 - h_f * hp_f * s5 * s7 + hm_f * hp_f * s10 * s8 - hm_f * s0 * s16 * s34 * (
                4 * df_f + s29) - 9 * hp_f * s0 * s21 * s28 + s0 * s13 * s3 + s0 * s14 * s5 + s0 * s15 * s22 * (
                        27 * s11 - 21 * s4 + s40) + s1 * s11 * s5 + 4 * s1 * s13 * s8 + s1 * s15 * s23 * (
                        -12 * s28 - 14 * s4 + s41 + 9 * s9) + s1 * s19 * s39 * (
                        s38 - s4 + s40) + 3 * s1 * s21 * s30 * (-dfp_f + s32) + s1 * s25 * s37 * (
                        s10 + s13 + s17 - 7 * s4 + s42) + 6 * s1 * s26 * s33 * (
                        dfm_f + dfp_f - s32) + s11 * s12 * s3 + s11 * s18 * s20 + s12 * s14 * s8 + 6 * s12 * s16 * s30 * (
                        df_f + s29) + s12 * s18 * s39 * (s13 + s38 - 9 * s4) + s12 * s22 * s23 * (
                        s24 - 42 * s4 + s41 + s42 + 3 * s9) + 13 * s13 * s18 * s25 + 12 * s13 * s22 * s26 + s14 * s20 * s25 + s15 * s16 * s17 + s15 * s18 * s30 * (
                        -7 * dfm_f + s31) - s15 * s25 * s36 * (
                        -8 * dfm_f + s31) + s15 * s26 * s27 + s16 * s23 * s24 + s17 * s21 * s22 - 4 * s18 * s23 * s33 * (
                        7 * df_f + s35) - s19 * s22 * s30 * (dfp_f - s31 + s34) - s19 * s23 * s36 * (
                        16 * df_f + s35) + s21 * s23 * s27 + s22 * s25 * s33 * (
                        -30 * df_f + 15 * dfm_f + s6) - s26 * s36 * s37 * (s32 + s35)) / (
                                 hm_f * s0 * s1 * s2 ** 2 * (h_f + hp_f) ** 2 * (hp_f + s2) ** 2)

        ### Now we compute the integral for the last interval.
        h_l = h[slice(-1, None)]
        hm_l = hm[slice(-1, None)]
        hp_l = hp[slice(-1, None)]
        df_l = df[slice(-1, None)]
        dfm_l = dfm[slice(-1, None)]
        dfp_l = dfp[slice(-1, None)]

        s0 = h_l ** 2
        s1 = hm_l ** 2
        s2 = h_l + hm_l
        s3 = hp_l ** 6
        s4 = 2 * dfm_l
        s5 = df_l * s4
        s6 = hm_l ** 6
        s7 = df_l * dfp_l
        s8 = h_l ** 6
        s9 = dfm_l * dfp_l
        s10 = 4 * s9
        s11 = df_l ** 2
        s12 = hp_l ** 2
        s13 = dfm_l ** 2
        s14 = dfp_l ** 2
        s15 = hm_l ** 3
        s16 = hp_l ** 5
        s17 = 3 * s11
        s18 = hm_l ** 4
        s19 = hp_l ** 4
        s20 = 4 * s19
        s21 = hm_l ** 5
        s22 = hp_l ** 3
        s23 = h_l ** 3
        s24 = 3 * s13
        s25 = h_l ** 4
        s26 = h_l ** 5
        s27 = 6 * s14
        s28 = df_l * dfm_l
        s29 = -dfp_l
        s30 = df_l * h_l
        s31 = 15 * df_l
        s32 = 2 * df_l
        s33 = dfp_l * hp_l
        s34 = 3 * dfp_l
        s35 = -s34
        s36 = dfm_l * hm_l
        s37 = 3 * s12
        s38 = 5 * s11
        s39 = 3 * s0
        s40 = -3 * s28 + s9
        s41 = 18 * s11
        s42 = -s5
        last_interval = 4 * (
                -h_l * hm_l * s3 * s5 - 2 * h_l * hp_l * s6 * s7 + hm_l * hp_l * s10 * s8 - 9 * hm_l * s0 * s16 * s28 - hp_l * s0 * s21 * s34 * (
                4 * df_l + s29) + s0 * s13 * s3 + s0 * s14 * s6 + s0 * s15 * s22 * (
                        27 * s11 + s40 - 21 * s7) + s1 * s11 * s3 + 4 * s1 * s14 * s8 + 3 * s1 * s16 * s30 * (
                        -dfm_l + s32) + s1 * s19 * s39 * (s38 + s40 - s7) + s1 * s22 * s23 * (
                        -12 * s28 + s41 - 14 * s7 + 9 * s9) + s1 * s25 * s37 * (
                        s10 + s14 + s17 + s42 - 7 * s7) + 6 * s1 * s26 * s33 * (
                        dfm_l + dfp_l - s32) + s11 * s12 * s6 + s11 * s18 * s20 + s12 * s13 * s8 + s12 * s15 * s23 * (
                        s27 + s41 + s42 - 42 * s7 + 3 * s9) + s12 * s18 * s39 * (
                        s14 + s38 - 9 * s7) + 6 * s12 * s21 * s30 * (
                        df_l + s29) + s13 * s20 * s25 + 12 * s14 * s15 * s26 + 13 * s14 * s18 * s25 + s15 * s16 * s17 - s15 * s19 * s30 * (
                        dfm_l - s31 + s34) + s15 * s25 * s33 * (
                        -30 * df_l + 15 * dfp_l + s4) + s16 * s23 * s24 + s17 * s21 * s22 + s18 * s22 * s30 * (
                        -7 * dfp_l + s31) - 4 * s18 * s23 * s33 * (7 * df_l + s35) - s19 * s23 * s36 * (
                        16 * df_l + s35) + s21 * s23 * s27 + s22 * s24 * s26 - s22 * s25 * s36 * (
                        -8 * dfp_l + s31) - s26 * s36 * s37 * (s32 + s35)) / (
                                hp_l * s0 * s1 * s2 ** 2 * (h_l + hp_l) ** 2 * (hp_l + s2) ** 2)

        ### Now, we stitch together the intervals.
        res = concatenate((
            first_interval,
            middle_intervals,
            last_interval,
        ))

        return res

    elif method in ["simpson"]:
        ### Forward Simpson for intervals 0 to N-2
        x2 = x[:-2]
        x3 = x[1:-1]
        x4 = x[2:]

        f2 = f[:-2]
        f3 = f[1:-1]
        f4 = f[2:]

        h = x3 - x2
        hp = x4 - x3

        df = f3 - f2
        dfp = f4 - f3

        res_forward_simpson = 4 * (df * hp - dfp * h) ** 2 / (h * hp ** 2 * (h + hp) ** 2)

        ### Backward Simpson for intervals 1 to N-1
        x1 = x[:-2]
        x2 = x[1:-1]
        x3 = x[2:]

        f1 = f[:-2]
        f2 = f[1:-1]
        f3 = f[2:]

        h = x3 - x2
        hm = x2 - x1

        dfm = f2 - f1
        df = f3 - f2

        res_backward_simpson = 4 * (df * hm - dfm * h) ** 2 / (h * hm ** 2 * (h + hm) ** 2)

        ### Fuse them together
        first_interval = res_forward_simpson[slice(0, 1)]

        a = res_backward_simpson[slice(None, -1)]
        b = res_forward_simpson[slice(1, None)]

        # middle_intervals = (a + b) / 2
        middle_intervals = ((a ** 2 + b ** 2) / 2 + 1e-100) ** 0.5  # This is more accurate across all frequencies

        last_interval = res_backward_simpson[slice(-1, None)]

        res = concatenate((
            first_interval,
            middle_intervals,
            last_interval,
        ))

        return res

    elif method in ["hybrid_simpson_cubic"]:
        from aerosandbox.numpy.calculus import gradient
        dfdx = gradient(
            f,
            x,
            edge_order=2
        )

        h = x[1:] - x[:-1]
        df = f[1:] - f[:-1]
        dfdx1 = dfdx[:-1]
        dfdx2 = dfdx[1:]

        res = (
                4 * (dfdx1 ** 2 + dfdx1 * dfdx2 + dfdx2 ** 2) / h
                + 12 * df / h ** 2 * (df / h - dfdx1 - dfdx2)
        )

        return res


    else:
        raise ValueError(f"Invalid method '{method}'.")


if __name__ == '__main__':
    import aerosandbox as asb
    import aerosandbox.numpy as np
    from scipy import integrate, interpolate
    import sympy as s

    np.random.seed(0)

    # degree = 4
    # coeffs = np.random.randn(degree + 1)
    # a = 1
    # b = 3
    #
    #
    # def f(x):
    #     out = 0
    #     for i in range(degree + 1):
    #         out += coeffs[i] * x ** i
    #
    #     return out

    a = 0
    b = 2


    def f(x):
        sin = np.sin if isinstance(x, np.ndarray) else s.sin
        return sin(2 * np.pi * x * 1) + 1


    print("\n\nTest 1: Integration")
    exact = integrate.quad(
        f,
        a,
        b,
        epsrel=1e-15,
    )[0]

    x_vals = np.cosspace(a, b, 100)
    f_vals = f(x_vals)

    approx_intervals = integrate_discrete_intervals(
        f=f_vals,
        x=x_vals,
        multiply_by_dx=True,
        # method="trapz",
        method="cubic",
        # method_endpoints="ignore",
    )

    integral = np.sum(approx_intervals)

    print(f"exact: {exact}")
    print(f"approx: {integral}")
    print(f"error: {integral - exact}")

    print("\n\nTest 2: Squared curvature")

    x_sym = s.symbols("x")
    f_sym = f(x_sym)
    df2dx_func = s.lambdify(x_sym, s.diff(f_sym, x_sym, 2))

    exact = integrate.quad(
        lambda x: df2dx_func(x) ** 2,
        a,
        b,
        epsrel=1e-15,
    )[0]
    print(f"exact: {exact}")

    approx = integrate_discrete_squared_curvature(
        f=f_vals,
        x=x_vals,
        # method="simpson"
        method="hybrid_simpson_cubic",
    )
    integral = np.sum(approx)
    print(f"\nintegrate_discrete_squared_curvature: {integral}")
    print(f"error: {integral - exact}")

    approx = integrate_discrete_intervals(
        f=np.gradient(
            f_vals,
            x_vals,
            n=2
        ) ** 2,
        x=x_vals,
    )
    integral = np.sum(approx)

    print(f"\nintegrate_discrete_intervals + np.gradient: {integral}")
    print(f"error: {integral - exact}")

    print("\n\nTest 3: Squared curvature with global-spline reconstruction")

    x = np.arange(0, 100 + 1)
    f = np.cos(np.pi * x / 2)

    f_interp = interpolate.InterpolatedUnivariateSpline(
        x=x,
        y=f,
        k=3
    )
    exact = integrate.quad(
        lambda x: f_interp.derivative(2)(x) ** 2,
        x[0],
        x[-1],
        epsrel=1e-8,
    )[0]

    approx = integrate_discrete_squared_curvature(
        f=f,
        x=x,
        method="hybrid_simpson_cubic",
        # method="simpson"
    )
    integral = np.sum(approx)

    print(f"exact: {exact}")
    print(f"\nintegrate_discrete_squared_curvature: {integral}")
    print(f"error: {integral - exact}")

    x_plot = np.linspace(x[0], x[-1], 10000)
    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    # p.qp(x_plot, f_interp.derivative(2)(x_plot) ** 2)
