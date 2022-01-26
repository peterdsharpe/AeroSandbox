import aerosandbox.numpy as np
from typing import Union, List


def linear_hermite_patch(
        x: Union[float, np.ndarray],
        x_a: float,
        x_b: float,
        f_a: float,
        f_b: float,
):
    return (x - x_a) * (f_b - f_a) / (x_b - x_a) + f_a


def cubic_hermite_patch(
        x: Union[float, np.ndarray],
        x_a: float,
        x_b: float,
        f_a: float,
        f_b: float,
        dfdx_a: float,
        dfdx_b: float,
):
    dx = x_b - x_a
    t = (x - x_a) / dx  # Nondimensional distance along the patch
    return (
            (t ** 3) * (1 * f_b) +
            (t ** 2 * (1 - t)) * (3 * f_b - 1 * dfdx_b * dx) +
            (t * (1 - t) ** 2) * (3 * f_a + 1 * dfdx_a * dx) +
            ((1 - t) ** 3) * (1 * f_a)
    )


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    x = np.linspace(0.8, 1)
    plt.plot(
        x,
        cubic_hermite_patch(
            x,
            x_a=x.min(),
            x_b=x.max(),
            f_a=0.002,
            f_b=0.5,
            dfdx_a=0.1,
            dfdx_b=10
        )
    )

    p.equal()
    p.show_plot()
