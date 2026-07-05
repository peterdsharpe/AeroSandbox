import aerosandbox.numpy as np
from aerosandbox.numpy.typing import Vectorizable
from typing import Literal


def linear_hermite_patch(
    x: Vectorizable,
    x_a: float,
    x_b: float,
    f_a: float,
    f_b: float,
) -> Vectorizable:
    """
    Compute the linear Hermite polynomial patch that passes through the given endpoints f_a and
    f_b.

    Parameters
    ----------
    x : Vectorizable
        Scalar or array of values at which to evaluate the patch.
    x_a : float
        The x-coordinate of the first endpoint.
    x_b : float
        The x-coordinate of the second endpoint.
    f_a : float
        The function value at the first endpoint.
    f_b : float
        The function value at the second endpoint.

    Returns
    -------
    Vectorizable
        The value of the patch evaluated at the input x. Returns a scalar if x is a scalar, or
        an array if x is an array.
    """
    return (x - x_a) * (f_b - f_a) / (x_b - x_a) + f_a


def cubic_hermite_patch(
    x: Vectorizable,
    x_a: float,
    x_b: float,
    f_a: float,
    f_b: float,
    dfdx_a: float,
    dfdx_b: float,
    extrapolation: Literal["continue", "clip"] = "continue",
) -> Vectorizable:
    """
    Compute the cubic Hermite polynomial patch that passes through the given endpoints and
    endpoint derivatives.

    Parameters
    ----------
    x : Vectorizable
        Scalar or array of values at which to evaluate the patch.
    x_a : float
        The x-coordinate of the first endpoint.
    x_b : float
        The x-coordinate of the second endpoint.
    f_a : float
        The function value at the first endpoint.
    f_b : float
        The function value at the second endpoint.
    dfdx_a : float
        The derivative of the function with respect to x at the first endpoint.
    dfdx_b : float
        The derivative of the function with respect to x at the second endpoint.
    extrapolation : Literal["continue", "clip"], optional
        A string indicating how to handle extrapolation outside of the domain [x_a, x_b]. Valid
        values are "continue", which continues the patch beyond the endpoints, and "clip", which
        clips the patch at the endpoints. Default is "continue".

    Returns
    -------
    Vectorizable
        The value of the patch evaluated at the input x. Returns a scalar if x is a scalar, or
        an array if x is an array.
    """
    dx = x_b - x_a
    t = (x - x_a) / dx  # Nondimensional distance along the patch
    if extrapolation == "continue":
        pass
    elif extrapolation == "clip":
        t = np.clip(t, 0, 1)
    else:
        raise ValueError(
            f"{extrapolation=!r} is not a valid option. "
            f"Valid options are: 'continue', 'clip'."
        )

    return (
        (t**3) * (1 * f_b)
        + (t**2 * (1 - t)) * (3 * f_b - 1 * dfdx_b * dx)
        + (t * (1 - t) ** 2) * (3 * f_a + 1 * dfdx_a * dx)
        + ((1 - t) ** 3) * (1 * f_a)
    )


def cosine_hermite_patch(
    x: Vectorizable,
    x_a: float,
    x_b: float,
    f_a: float,
    f_b: float,
    dfdx_a: float,
    dfdx_b: float,
    extrapolation: Literal["continue", "linear"] = "continue",
) -> Vectorizable:
    r"""
    Compute a Hermite patch (i.e., values + derivatives at endpoints) that uses a cosine
    function to blend between linear segments.

    The end result is conceptually similar to a cubic Hermite patch, but computation is faster
    and the patch is $C^\infty$-continuous.

    Parameters
    ----------
    x : Vectorizable
        Scalar or array of values at which to evaluate the patch.
    x_a : float
        The x-coordinate of the first endpoint.
    x_b : float
        The x-coordinate of the second endpoint.
    f_a : float
        The function value at the first endpoint.
    f_b : float
        The function value at the second endpoint.
    dfdx_a : float
        The derivative of the function with respect to x at the first endpoint.
    dfdx_b : float
        The derivative of the function with respect to x at the second endpoint.
    extrapolation : Literal["continue", "linear"], optional
        A string indicating how to handle extrapolation outside of the domain [x_a, x_b]. Valid
        values are "continue", which continues the patch beyond the endpoints, and "linear",
        which extends the patch linearly at the endpoints. Default is "continue".

    Returns
    -------
    Vectorizable
        The value of the patch evaluated at the input x. Returns a scalar if x is a scalar, or
        an array if x is an array.
    """
    t = (x - x_a) / (x_b - x_a)  # Nondimensional distance along the patch
    if extrapolation == "continue":
        pass
    elif extrapolation == "linear":
        t = np.clip(t, 0, 1)
    else:
        raise ValueError(
            f"{extrapolation=!r} is not a valid option. "
            f"Valid options are: 'continue', 'linear'."
        )

    l1 = (x - x_a) * dfdx_a + f_a
    l2 = (x - x_b) * dfdx_b + f_b

    b = 0.5 + 0.5 * np.cos(np.pi * t)

    return b * l1 + (1 - b) * l2


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    x = np.linspace(-1, 2, 500)
    plt.plot(
        x,
        cubic_hermite_patch(
            x, x_a=0, x_b=1, f_a=0, f_b=1, dfdx_a=-0.5, dfdx_b=-1, extrapolation="clip"
        ),
    )

    p.equal()
    p.show_plot()
