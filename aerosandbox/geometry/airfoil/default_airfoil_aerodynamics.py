import aerosandbox.numpy as np
import warnings
from aerosandbox.library.aerodynamics.viscous import Cf_flat_plate

def print_default_warning():
    warnings.warn("Warning: Using a default flat-plate aerodynamics model for this airfoil!\n"
                  "To use a better, more accurate one, specify functions in the Airfoil constructor.")


def default_CL_function(alpha, Re, mach, deflection):
    """
    Lift coefficient.
    """
    print_default_warning()
    Cl_inc = 2 * np.pi * np.radians(alpha)
    beta = (1 - mach) ** 2

    Cl = Cl_inc * beta
    return Cl


def default_CD_function(alpha, Re, mach, deflection):
    """
    Drag coefficient.
    """
    print_default_warning()
    Cf = Cf_flat_plate(Re_L = Re)
    Cd_inc = 2 * Cf * (
            1 + (alpha / 5) ** 2
    )
    beta = (1 - mach) ** 2

    Cd = Cd_inc * beta
    return Cd


def default_CM_function(alpha, Re, mach, deflection):
    """
    Pitching moment coefficient, as measured about quarter-chord.
    """
    print_default_warning()
    return 0
