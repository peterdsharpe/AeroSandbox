### This file contains an assortment of random airfoils to use
from aerosandbox.geometry.airfoil import Airfoil
from aerosandbox.library.aerodynamics.viscous import *
from aerosandbox.geometry.airfoil.airfoil_families import get_NACA_coordinates,\
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
    y = y * t_over_c

    coordinates = np.array([x, y]).T

    return Airfoil(
        name="Diamond",
        coordinates=coordinates,
    )


generic_cambered_airfoil = Airfoil(
    name="Generic Cambered Airfoil",
    CL_function=lambda alpha, Re, mach, deflection,: (  # Lift coefficient function
            (alpha * np.pi / 180) * (2 * np.pi) + 0.4550
    ),
    CD_function=lambda alpha, Re, mach, deflection: (  # Profile drag coefficient function
            (1 + (alpha / 5) ** 2) * 2 * Cf_flat_plate(Re_L=Re)
    ),
    CM_function=lambda alpha, Re, mach, deflection: (  # Moment coefficient function about quarter-chord
        -0.1
    ),
    coordinates=get_UIUC_coordinates(name="clarky")
)
generic_airfoil = Airfoil(
    name="Generic Airfoil",
    CL_function=lambda alpha, Re, mach, deflection,: (  # Lift coefficient function
            (alpha * np.pi / 180) * (2 * np.pi)
    ),
    CD_function=lambda alpha, Re, mach, deflection: (  # Profile drag coefficient function
            (1 + (alpha / 5) ** 2) * 2 * Cf_flat_plate(Re_L=Re)
    ),
    CM_function=lambda alpha, Re, mach, deflection: (  # Moment coefficient function about quarter-chord
        0
    ), # TODO make this an actual curve!
    coordinates=get_NACA_coordinates(name="naca0012")
)

e216 = Airfoil(
    name="e216",
    CL_function=lambda alpha, Re, mach, deflection: (  # Lift coefficient function
        Cl_e216(alpha=alpha, Re_c=Re)
    ),
    CD_function=lambda alpha, Re, mach, deflection: (  # Profile drag coefficient function
            Cd_profile_e216(alpha=alpha, Re_c=Re) +
            Cd_wave_e216(Cl=Cl_e216(alpha=alpha, Re_c=Re), mach=mach)
    ),
    CM_function=lambda alpha, Re, mach, deflection: (  # Moment coefficient function about quarter-chord
        -0.15
    ),  # TODO make this an actual curve!
)

rae2822 = Airfoil(
    name="rae2822",
    CL_function=lambda alpha, Re, mach, deflection: (  # Lift coefficient function
        Cl_rae2822(alpha=alpha, Re_c=Re)
    ),
    CD_function=lambda alpha, Re, mach, deflection: (  # Profile drag coefficient function
            Cd_profile_rae2822(alpha=alpha, Re_c=Re) +
            Cd_wave_rae2822(Cl=Cl_rae2822(alpha=alpha, Re_c=Re), mach=mach)
    ),
    CM_function=lambda alpha, Re, mach, deflection: (  # Moment coefficient function about quarter-chord
        -0.05
    ),  # TODO make this an actual curve!
)

naca0008 = Airfoil(
    name="naca0008",
    CL_function=lambda alpha, Re, mach, deflection: (  # Lift coefficient function
        Cl_flat_plate(alpha=alpha)  # TODO fit this to actual data
    ),
    CD_function=lambda alpha, Re, mach, deflection: (  # Profile drag coefficient function
            (1 + (alpha / 5) ** 2) * 2 * Cf_flat_plate(Re_L=Re) +  # TODO fit this to actual data
            Cd_wave_Korn(Cl=Cl_flat_plate(alpha=alpha), t_over_c=0.08, mach=mach, kappa_A=0.87)
    ),
    CM_function=lambda alpha, Re, mach, deflection: (  # Moment coefficient function about quarter-chord
        0
    ),  # TODO make this an actual curve!
)

flat_plate = Airfoil(
    name="Flat Plate",
    CL_function=lambda alpha, Re, mach, deflection,: (  # Lift coefficient function
        Cl_flat_plate(alpha=alpha)
    ),
    CD_function=lambda alpha, Re, mach, deflection,: (  # Profile drag coefficient function
            Cf_flat_plate(Re_L=Re) * 2
    ),
    CM_function=lambda alpha, Re, mach, deflection: (  # Moment coefficient function
        0
    ),
    coordinates=np.array([
        [1, 0],
        [1, 1e-6],
        [0, 1e-6],
        [0, -1e-6],
        [1, -1e-6],
        [1, 0],
        ])
)
