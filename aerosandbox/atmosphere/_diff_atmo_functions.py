import aerosandbox.numpy as np
from pathlib import Path
from aerosandbox.modeling.interpolation import InterpolatedModel
from aerosandbox.atmosphere._isa_atmo_functions import pressure_isa, temperature_isa, isa_base_altitude

# Define the altitudes of knot points
# altitude_knot_points = np.concatenate((
#     np.linspace(-50e3, 9e3, 50),
#     np.linspace(13e3, 18e3, 10),
#     np.linspace(22e3, 30e3, 10),
#     np.linspace(34e3, 45e3, 10),
#     np.linspace(48e3, 50e3, 10),
#     np.linspace(53e3, 69e3, 10),
#     np.linspace(73e3, 85e3, 10),
#     np.linspace(89e3, 150e3, 40)
# ))

altitude_knot_points = np.array(
    list(isa_base_altitude - 2e3) +
    list(isa_base_altitude + 2e3) +
    list(isa_base_altitude[-1] + np.geomspace(5e3, 1000e3, 11)) +
    list(isa_base_altitude[0] - np.geomspace(5e3, 5000e3, 11))
)

altitude_knot_points = np.sort(np.unique(altitude_knot_points))

temperature_knot_points = temperature_isa(altitude_knot_points)
pressure_knot_points = pressure_isa(altitude_knot_points)

# creates interpolated model for temperature and pressure
interpolated_temperature = InterpolatedModel(
    x_data_coordinates=altitude_knot_points,
    y_data_structured=temperature_knot_points,
)
interpolated_log_pressure = InterpolatedModel(
    x_data_coordinates=altitude_knot_points,
    y_data_structured=np.log(pressure_knot_points),
)


def pressure_differentiable(altitude):
    """
    Computes the pressure at a given altitude with a differentiable model.

    Args:
        altitude: Geopotential altitude [m]

    Returns: Pressure [Pa]

    """
    return np.exp(interpolated_log_pressure(altitude))


def temperature_differentiable(altitude):
    """
    Computes the temperature at a given altitude with a differentiable model.

    Args:
        altitude: Geopotential altitude [m]

    Returns: Temperature [K]

    """
    return interpolated_temperature(altitude)
