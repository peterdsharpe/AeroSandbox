### This file contains an assortment of random airfoils to use
from aerosandbox.geometry.airfoil import Airfoil
import aerosandbox.numpy as np


def diamond_airfoil(
    t_over_c: float,
    n_points_per_panel=2,
) -> Airfoil:
    """
    Create a diamond (double-wedge) airfoil with a specified thickness-to-chord ratio.

    Parameters
    ----------
    t_over_c : float
        The thickness-to-chord ratio of the airfoil. [nondimensional]
    n_points_per_panel
        The number of coordinate points per face of the diamond.

    Returns
    -------
    Airfoil
        The diamond airfoil, as an AeroSandbox Airfoil object.
    """
    x_nondim = [1, 0.5, 0, 0.5, 1]
    y_nondim = [0, 1, 0, -1, 0]

    x = np.concatenate(
        [
            list(np.cosspace(a, b, n_points_per_panel))[:-1]
            for a, b in zip(x_nondim[:-1], x_nondim[1:])
        ]
        + [[x_nondim[-1]]]
    )
    y = np.concatenate(
        [
            list(np.cosspace(a, b, n_points_per_panel))[:-1]
            for a, b in zip(y_nondim[:-1], y_nondim[1:])
        ]
        + [[y_nondim[-1]]]
    )
    y = y * (t_over_c / 2)

    coordinates = np.array([x, y]).T

    return Airfoil(
        name="Diamond",
        coordinates=coordinates,
    )
