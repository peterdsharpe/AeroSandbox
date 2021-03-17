from aerosandbox import Atmosphere
import pytest
import aerosandbox.numpy as np
import pandas as pd
from pathlib import Path

"""
Validates the Atmosphere class against data from the International Standard Atmosphere (ISA).

Some deviation is allowed, as the ISA model is not C1-continuous, but we want our to be C1-continuous for optimization. 
"""

isa_data = pd.read_csv(str(
    Path(__file__).parent / "../isa_data/isa_sample_values.csv"
))
altitudes = isa_data["Altitude [m]"].values
pressures = isa_data["Pressure [Pa]"].values
temperatures = isa_data["Temperature [K]"].values
densities = isa_data["Density [kg/m^3]"].values
speeds_of_sound = isa_data["Speed of Sound [m/s]"].values


def test_isa_atmosphere():
    for altitude, pressure, temperature, density, speed_of_sound in zip(
            altitudes,
            pressures,
            temperatures,
            densities,
            speeds_of_sound
    ):

        atmo = Atmosphere(altitude=altitude, method='isa')

        if altitude >= atmo._valid_altitude_range[0] and altitude <= atmo._valid_altitude_range[1]:

            fail_message = f"FAILED @ {altitude} m"

            assert atmo.pressure() == pytest.approx(pressure, abs=100), fail_message
            assert atmo.temperature() == pytest.approx(temperature, abs=1), fail_message
            assert atmo.density() == pytest.approx(density, abs=0.01), fail_message
            assert atmo.speed_of_sound() == pytest.approx(speed_of_sound, abs=1), fail_message


def test_diff_atmosphere():
    altitudes = np.linspace(-50e2, 150e3, 1000)
    atmo_isa = Atmosphere(altitude=altitudes, method='isa')
    atmo_diff = Atmosphere(altitude=altitudes)
    temp_isa = atmo_isa.temperature()
    pressure_isa = atmo_isa.pressure()
    temp_diff = atmo_diff.temperature()
    pressure_diff = atmo_diff.pressure()
    assert max(abs((temp_isa - temp_diff) / temp_isa)) < 0.025, "temperature failed for differentiable model"
    assert max(abs((pressure_isa - pressure_diff) / pressure_isa)) < 0.01, "pressure failed for differentiable model"


def plot_isa_residuals():
    atmo = Atmosphere(altitude=altitudes, method='isa')

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(palette=sns.color_palette("husl"))

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
    plt.plot(altitudes, atmo.pressure() - pressures)
    plt.xlabel(r"Altitude [m]")
    plt.ylabel(r"Pressure Difference [Pa]")
    plt.title(r"Pressure Difference between ASB ISA Model and ISA Table")
    plt.tight_layout()
    plt.legend()
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
    plt.plot(altitudes, atmo.temperature() - temperatures)
    plt.xlabel(r"Altitude [m]")
    plt.ylabel(r"Temperature Difference [K]")
    plt.title(r"Temperature Difference between ASB ISA Model and ISA Table")
    plt.tight_layout()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # plot_isa_residuals()
    pytest.main()
