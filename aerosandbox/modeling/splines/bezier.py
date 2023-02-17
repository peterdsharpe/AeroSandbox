import aerosandbox.numpy as np
from typing import Union, Tuple


def quadratic_bezier_patch_from_tangents(
        t: Union[float, np.ndarray],
        x_a: float,
        x_b: float,
        y_a: float,
        y_b: float,
        dydx_a: float,
        dydx_b: float,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Computes sampled points in 2D space from a quadratic Bezier spline defined by endpoints and end-tangents.

    Note: due to the inherent nature of a quadratic Bezier curve, curvature will be strictly one-sided - in other
    words, this will not make "S"-shaped curves. This means that you should be aware that bad values of dydx at
    either endpoint might cause this curvature to flip, which would result in the curve "going backwards" at one
    endpoint.

    Also, note that, in general, points will not be spaced evenly in x, y, or arc length s.

    Args:
        t:
        x_a:
        x_b:
        y_a:
        y_b:
        dydx_a:
        dydx_b:

    Returns:

    """
    ### Compute intercept of tangent lines
    x_P1 = (
                   (y_b - y_a) + (dydx_a * x_a - dydx_b * x_b)
           ) / (dydx_a - dydx_b)
    y_P1 = y_a + dydx_a * (x_P1 - x_a)

    x = (
            (1 - t) ** 2 * x_a +
            2 * (1 - t) * t * x_P1 +
            t ** 2 * x_b
    )
    y = (
            (1 - t) ** 2 * y_a +
            2 * (1 - t) * t * y_P1 +
            t ** 2 * y_b
    )

    return x, y

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    x, y = quadratic_bezier_patch_from_tangents(
        t = np.linspace(0, 1, 11),
        x_a=1,
        x_b=4,
        y_a=2,
        y_b=3,
        dydx_a=1,
        dydx_b=-30
    )

    plt.plot(x, y, ".-")
    p.show_plot()