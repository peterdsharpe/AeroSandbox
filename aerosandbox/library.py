### This file contains an assortment of random objects to use
from .geometry import *

generic_cambered_airfoil = Airfoil(
    CL_function=lambda alpha, Re, mach, deflection,: (  # Lift coefficient function
            (alpha * np.pi / 180) * (2 * np.pi) + 0.4550
    ),
    CDp_function=lambda alpha, Re, mach, deflection: (  # Profile drag coefficient function
            (1 + (alpha / 5) ** 2) * 2 * (0.074 / Re ** 0.2)
    ),
    Cm_function=lambda alpha, Re, mach, deflection: (  # Moment coefficient function
        0
    )
)
generic_airfoil = Airfoil(
    CL_function=lambda alpha, Re, mach, deflection,: (  # Lift coefficient function
            (alpha * np.pi / 180) * (2 * np.pi)
    ),
    CDp_function=lambda alpha, Re, mach, deflection: (  # Profile drag coefficient function
            (1 + (alpha / 5) ** 2) * 2 * (0.074 / Re ** 0.2)
    ),
    Cm_function=lambda alpha, Re, mach, deflection: (  # Moment coefficient function
        0
    )
)