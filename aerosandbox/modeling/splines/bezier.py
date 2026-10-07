import aerosandbox.numpy as np
from aerosandbox.numpy.typing import Vectorizable


def quadratic_bezier_patch_from_tangents(
    t: Vectorizable,
    x_a: float,
    x_b: float,
    y_a: float,
    y_b: float,
    dydx_a: float,
    dydx_b: float,
) -> tuple[Vectorizable, Vectorizable]:
    """
    Compute sampled points in 2D space from a quadratic Bezier spline defined by endpoints and
    end-tangents.

    Note: due to the inherent nature of a quadratic Bezier curve, curvature will be strictly
    one-sided - in other words, this will not make "S"-shaped curves. This means that you should
    be aware that bad values of dydx at either endpoint might cause this curvature to flip,
    which would result in the curve "going backwards" at one endpoint.

    Note also that the end tangents may not be parallel (i.e., `dydx_a != dydx_b` is required),
    since the curve's control point is placed at the intersection of the two end-tangent lines.
    This includes the straight-line case `dydx_a == dydx_b == (y_b - y_a) / (x_b - x_a)`; for
    that case, interpolate between the endpoints directly instead.

    Also, note that, in general, points will not be spaced evenly in x, y, or arc length s.

    Parameters
    ----------
    t : Vectorizable
        The nondimensional parameter values at which to sample the curve, typically in the
        range [0, 1]. (t=0 gives the first endpoint; t=1 gives the second endpoint.)
    x_a : float
        The x-coordinate of the first endpoint.
    x_b : float
        The x-coordinate of the second endpoint.
    y_a : float
        The y-coordinate of the first endpoint.
    y_b : float
        The y-coordinate of the second endpoint.
    dydx_a : float
        The derivative of y with respect to x at the first endpoint.
    dydx_b : float
        The derivative of y with respect to x at the second endpoint.

    Returns
    -------
    x : Vectorizable
        A scalar or numpy array of scalars representing the x-coordinates of the sampled points.
    y : Vectorizable
        A scalar or numpy array of scalars representing the y-coordinates of the sampled points.

    Examples
    --------
    >>> x_a, x_b = 0, 10
    >>> y_a, y_b = 0, 5
    >>> dydx_a, dydx_b = 0.5, -0.5
    >>>
    >>> t = np.linspace(0, 1, 50)
    >>> x, y = quadratic_bezier_patch_from_tangents(
    >>>     t=t,
    >>>     x_a=x_a,
    >>>     x_b=x_b,
    >>>     y_a=y_a,
    >>>     y_b=y_b,
    >>>     dydx_a=dydx_a,
    >>>     dydx_b=dydx_b
    >>> )
    """
    ### Guard against parallel end tangents, which would put the intercept at infinity (division by zero).
    ### (Only checkable for numeric inputs; symbolic CasADi inputs pass through unchecked.)
    if not (
        np.is_casadi_type(dydx_a, recursive=False)
        or np.is_casadi_type(dydx_b, recursive=False)
    ):
        if np.any(dydx_a == dydx_b):
            raise ValueError(
                "The end tangents `dydx_a` and `dydx_b` may not be parallel (equal), since the quadratic Bezier "
                "control point is placed at the intersection of the two end-tangent lines. "
                "For the straight-line case, interpolate between the endpoints directly instead."
            )

    ### Compute intercept of tangent lines
    x_P1 = ((y_b - y_a) + (dydx_a * x_a - dydx_b * x_b)) / (dydx_a - dydx_b)
    y_P1 = y_a + dydx_a * (x_P1 - x_a)

    x = (1 - t) ** 2 * x_a + 2 * (1 - t) * t * x_P1 + t**2 * x_b
    y = (1 - t) ** 2 * y_a + 2 * (1 - t) * t * y_P1 + t**2 * y_b

    return x, y


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    x, y = quadratic_bezier_patch_from_tangents(
        t=np.linspace(0, 1, 11), x_a=1, x_b=4, y_a=2, y_b=3, dydx_a=1, dydx_b=-30
    )

    plt.plot(x, y, ".-")
    p.show_plot()
