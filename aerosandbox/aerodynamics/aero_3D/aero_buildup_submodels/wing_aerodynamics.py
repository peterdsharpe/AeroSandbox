from aerosandbox.geometry import *
from aerosandbox.performance import OperatingPoint
import aerosandbox.numpy as np
from aerosandbox.library.aerodynamics import Cf_flat_plate, Cd_cylinder


def wing_aerodynamics(
        wing: Wing,
        op_point: OperatingPoint,
):
    """
    Estimates the aerodynamic forces, moments, and derivatives on a wing in isolation.

    Assumes:
        * The fuselage is a body of revolution aligned with the x_b axis.
        * The angle between the nose and the freestream is less than 90 degrees.

    Moments are given with the reference at Wing [0, 0, 0].

    Args:

        fuselage: A Wing object that you wish to analyze.

        op_point: The OperatingPoint that you wish to analyze the fuselage at.

    Returns:

    """
    pass
