from aerosandbox import Atmosphere
import pytest
import pandas as pd
from pathlib import Path

"""
Validates the Atmosphere class against data from the International Standard Atmosphere (ISA).

Some deviation is allowed, as the ISA model is not C1-continuous, but we want our to be C1-continuous for optimization. 

All test values pulled from International Standard Atmosphere (ISA) tables.
https://www.engineeringtoolbox.com/international-standard-atmosphere-d_985.html
"""

isa_data = pd.read_csv(str(
    Path(__file__).parent.absolute() / "isa_data.csv"
))


def approx(x):
    """
    Creates an approximator like pytest.approx, but with the tolerance baked in.
    """
    return pytest.approx(x, rel=0.02)


def test_atmosphere():
    for altitude, pressure, temperature, density, speed_of_sound in zip(
            isa_data["Altitude [m]"],
            isa_data["Pressure [Pa]"],
            isa_data["Temperature [K]"],
            isa_data["Density [kg/m^3]"],
            isa_data["Speed of Sound [m/s]"]
    ):

        atmo = Atmosphere(altitude=altitude)

        if altitude > atmo._valid_range[0] and altitude < atmo._valid_range[1]:

            fail_message = f"{altitude} m"

            assert atmo.pressure() == approx(pressure), fail_message
            assert atmo.temperature() == approx(temperature), fail_message
            assert atmo.density() == approx(density), fail_message
            assert atmo.speed_of_sound() == approx(speed_of_sound), fail_message


if __name__ == '__main__':
    pytest.main()
