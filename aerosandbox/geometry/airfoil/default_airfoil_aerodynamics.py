import aerosandbox.numpy as np
import warnings
from aerosandbox.library.aerodynamics.viscous import Cf_flat_plate


warnings.warn(
    "This file is deprecated.\n"
    "You can use `asb.Airfoil.get_aero_from_neuralfoil()` instead\n"
    "to get airfoil aerodynamics for any airfoil.",
    DeprecationWarning
)

def print_default_warning():
    warnings.warn("\n".join([
        "Warning: Using a placeholder aerodynamics model for this Airfoil!",
        "It's highly recommended that you either:",
        "\ta) Specify polar functions in the Airfoil constructor, or",
        "\tb) Call Airfoil.generate_polars() to auto-generate these polar functions with XFoil."
    ]), stacklevel=3)


def default_CL_function(alpha, Re, mach=0, deflection=0):
    """
    Lift coefficient.
    """
    print_default_warning()
    Cl_inc = np.pi * np.sind(2 * alpha)
    beta = (1 - mach) ** 2

    Cl = Cl_inc * beta
    return Cl


def default_CD_function(alpha, Re, mach=0, deflection=0):
    """
    Drag coefficient.
    """
    print_default_warning()
    Cf = Cf_flat_plate(Re_L=Re, method="hybrid-sharpe-convex")

    ### Form factor model from Raymer, "Aircraft Design". Section 12.5, Eq. 12.30
    t_over_c = 0.12
    FF = 1 + 2 * t_over_c * 100 * t_over_c ** 4

    Cd_inc = 2 * Cf * FF * (
            1 + (np.sind(alpha) * 180 / np.pi / 5) ** 2
    )
    beta = (1 - mach) ** 2

    Cd = Cd_inc * beta
    return Cd


def default_CM_function(alpha, Re, mach=0, deflection=0):
    """
    Pitching moment coefficient, as measured about quarter-chord.
    """
    print_default_warning()
    return np.zeros_like(alpha)
