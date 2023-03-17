import aerosandbox as asb
import aerosandbox.numpy as np
import aerosandbox.tools.units as u
from typing import Dict, Union

"""
Applicable to:

* low-to-moderate performance (up to about 300 kt) light utility aircraft.
"""


def mass_landing_gear(
        gear_length: float,
        design_mass_TOGW: float,
        ultimate_load_factor: float,
):
    """
    Calculates the mass of the landing gear.

    Args:

        gear_length: The length of the landing gear, in meters.

        design_mass_TOGW: The design takeoff gross weight of the aircraft, in kg.

        ultimate_load_factor: The ultimate load factor of the aircraft.

    Returns: The mass of the landing gear, in kg.
    """
    return (
            0.054 *
            (gear_length / u.inch) ** 0.501 *
            (design_mass_TOGW / u.lbm * ultimate_load_factor) ** 0.684
    ) * u.lbm
