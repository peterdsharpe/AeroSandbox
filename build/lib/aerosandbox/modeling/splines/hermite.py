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
        extrapolation: str = 'continue',
):
    dx = x_b - x_a
    t = (x - x_a) / dx  # Nondimensional distance along the patch
    if extrapolation == 'continue':
        pass
    elif extrapolation == 'clip':
        t = np.clip(t, 0, 1)
    else:
        raise ValueError("Bad value of `extrapolation`!")

    return (
            (t ** 3) * (1 * f_b) +
            (t ** 2 * (1 - t)) * (3 * f_b - 1 * dfdx_b * dx) +
            (t * (1 - t) ** 2) * (3 * f_a + 1 * dfdx_a * dx) +
            ((1 - t) ** 3) * (1 * f_a)
    )


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    x = np.linspace(-1, 2, 500)
    plt.plot(
        x,
        cubic_hermite_patch(
            x,
            x_a=0,
            x_b=1,
            f_a=0,
            f_b=1,
            dfdx_a=-0.5,
            dfdx_b=-1,
            extrapolation='clip'
        )
    )

    p.equal()
    p.show_plot()
