import casadi as cas
import numpy as np
from aerosandbox.tools.casadi_tools import *


def CL_over_Cl(AR, mach=0, sweep=0):
    """
    Returns the ratio of 3D lift coefficient (with compressibility) to 2D lift coefficient (incompressible).
    :param AR: Aspect ratio
    :param mach: Mach number
    :param sweep: Sweep angle [deg]
    :return:
    """
    beta = cas.if_else(
        1 - mach ** 2 >= 0,
        cas.sqrt(1 - mach ** 2),
        0
    )
    sweep_rad = sweep * np.pi / 180
    # return AR / (AR + 2) # Equivalent to equation in Drela's FVA in incompressible, 2*pi*alpha limit.
    # return AR / (2 + cas.sqrt(4 + AR ** 2))  # more theoretically sound at low AR
    eta = 0.95
    return AR / (
            2 + cas.sqrt(
        4 + (AR * beta / eta) ** 2 * (1 + (cas.tan(sweep_rad) / beta) ** 2)
    )
    )  # From Raymer, Sect. 12.4.1; citing DATCOM


def induced_drag_ratio_from_ground_effect(
        h_over_b  # type: float
):
    """
    Gives the ratio of actual induced drag to free-flight induced drag experienced by a wing in ground effect.
    Artificially smoothed below around h/b == 0.05 to retain differentiability and practicality.
    Source: W. F. Phillips, D. F. Hunsaker, "Lifting-Line Predictions for Induced Drag and Lift in Ground Effect".
        Using Equation 5 from the paper, which is modified from a model from Torenbeek:
            Torenbeek, E. "Ground Effects", 1982.
    :param h_over_b: (Height above ground) divided by (wingspan).
    :return: Ratio of induced drag in ground effect to induced drag out of ground effect [unitless]
    """
    h_over_b = smoothmax(
        h_over_b,
        0,
        hardness=1 / 0.03
    )
    return 1 - cas.exp(
        -4.01 * (2 * h_over_b) ** 0.717
    )
