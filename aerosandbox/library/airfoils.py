### This file contains an assortment of random airfoils to use
from aerosandbox.geometry.airfoil import Airfoil
from aerosandbox.library.aerodynamics.viscous import *

generic_cambered_airfoil = Airfoil(
    name="Generic Cambered Airfoil",
    coordinates=None,
    CL_function=lambda alpha, Re, mach, deflection,: (  # Lift coefficient function
            (alpha * np.pi / 180) * (2 * np.pi) + 0.4550
    ),
    CD_function=lambda alpha, Re, mach, deflection: (  # Profile drag coefficient function
            (1 + (alpha / 5) ** 2) * 2 * Cf_flat_plate(Re_L=Re)
    ),
    CM_function=lambda alpha, Re, mach, deflection: (  # Moment coefficient function about quarter-chord
        -0.1
    )
)
generic_airfoil = Airfoil(
    name="Generic Airfoil",
    coordinates=None,
    CL_function=lambda alpha, Re, mach, deflection,: (  # Lift coefficient function
            (alpha * np.pi / 180) * (2 * np.pi)
    ),
    CD_function=lambda alpha, Re, mach, deflection: (  # Profile drag coefficient function
            (1 + (alpha / 5) ** 2) * 2 * Cf_flat_plate(Re_L=Re)
    ),
    CM_function=lambda alpha, Re, mach, deflection: (  # Moment coefficient function about quarter-chord
        0
    )  # TODO make this an actual curve!
)

e216 = Airfoil(
    name="e216",
    coordinates=None,
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
    coordinates=None,
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
    coordinates=None,
    CL_function=lambda alpha, Re, mach, deflection: (  # Lift coefficient function
        Cl_flat_plate(alpha=alpha, Re_c=Re)  # TODO fit this to actual data
    ),
    CD_function=lambda alpha, Re, mach, deflection: (  # Profile drag coefficient function
            (1 + (alpha / 5) ** 2) * 2 * Cf_flat_plate(Re_L=Re) +  # TODO fit this to actual data
            Cd_wave_Korn(Cl=Cl_flat_plate(alpha=alpha, Re_c=Re), t_over_c=0.08, mach=mach, kappa_A=0.87)
    ),
    CM_function=lambda alpha, Re, mach, deflection: (  # Moment coefficient function about quarter-chord
        0
    ),  # TODO make this an actual curve!
)

flat_plate = Airfoil(
    name="Flat Plate",
    coordinates=None,
    CL_function=lambda alpha, Re, mach, deflection,: (  # Lift coefficient function
        Cl_flat_plate(alpha=alpha, Re_c=Re)
    ),
    CD_function=lambda alpha, Re, mach, deflection,: (  # Profile drag coefficient function
            Cf_flat_plate(Re_L=Re) * 2
    ),
    CM_function=lambda alpha, Re, mach, deflection: (  # Moment coefficient function
        0
    )  # TODO make this an actual curve!
)
