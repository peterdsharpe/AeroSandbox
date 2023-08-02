### This file contains an assortment of random airfoils to use
from aerosandbox.geometry.airfoil import Airfoil
from aerosandbox.library.aerodynamics.viscous import *
from aerosandbox.geometry.airfoil.airfoil_families import get_NACA_coordinates, \
    get_UIUC_coordinates


def diamond_airfoil(
        t_over_c: float,
        n_points_per_panel=2,
) -> Airfoil:
    x_nondim = [1, 0.5, 0, 0.5, 1]
    y_nondim = [0, 1, 0, -1, 0]

    x = np.concatenate(
        [
            list(np.cosspace(a, b, n_points_per_panel))[:-1]
            for a, b in zip(x_nondim[:-1], x_nondim[1:])
        ] + [[x_nondim[-1]]]
    )
    y = np.concatenate(
        [
            list(np.cosspace(a, b, n_points_per_panel))[:-1]
            for a, b in zip(y_nondim[:-1], y_nondim[1:])
        ] + [[y_nondim[-1]]]
    )
    y = y * (t_over_c / 2)

    coordinates = np.array([x, y]).T

    return Airfoil(
        name="Diamond",
        coordinates=coordinates,
    )

