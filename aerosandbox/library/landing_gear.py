import aerosandbox as asb
import aerosandbox.numpy as np
import aerosandbox.tools.units as u


def tire_size(
        mass_supported_by_each_tire: float,
        aircraft_type="general_aviation"
) -> float:
    """
    Computes the required diameter and width of a tire for an airplane, from statistical regression to historical data.

    Methodology and constants from Raymer: Aircraft Design: A Conceptual Approach, 5th Edition, Table 11.1, pg. 358.

    Args:
        mass_supported_by_each_tire: The mass supported by each tire, in kg.

        aircraft_type: The type of aircraft. Options are:
            - "general_aviation"
            - "business_twin"
            - "transport/bomber"
            - "fighter/trainer"

    Returns:
        The required diameter and width of the tire, in meters.
    """
    mass_supported_by_tire_lbm = mass_supported_by_each_tire / u.lbm

    if aircraft_type == "general_aviation":
        A = 1.51
        B = 0.349
    elif aircraft_type == "business_twin":
        A = 2.69
        B = 0.251
    elif aircraft_type == "transport/bomber":
        A = 1.63
        B = 0.315
    elif aircraft_type == "fighter/trainer":
        A = 1.59
        B = 0.302
    else:
        raise ValueError("Invalid `aircraft_type`.")

    tire_diameter_in = A * mass_supported_by_tire_lbm ** B

    if aircraft_type == "general_aviation":
        A = 0.7150
        B = 0.312
    elif aircraft_type == "business_twin":
        A = 1.170
        B = 0.216
    elif aircraft_type == "transport/bomber":
        A = 0.1043
        B = 0.480
    elif aircraft_type == "fighter/trainer":
        A = 0.0980
        B = 0.467

    tire_width_in = A * mass_supported_by_tire_lbm ** B

    tire_diameter = tire_diameter_in * u.inch
    tire_width = tire_width_in * u.inch

    return tire_diameter, tire_width
