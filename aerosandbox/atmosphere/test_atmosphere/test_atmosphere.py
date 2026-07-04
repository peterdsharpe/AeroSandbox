from aerosandbox import Atmosphere
import pytest
import aerosandbox.numpy as np
import pandas as pd
from pathlib import Path

"""
Validates the Atmosphere class against data from the International Standard Atmosphere (ISA).

Some deviation is allowed, as the ISA model is not C1-continuous, but we want our to be C1-continuous for optimization. 
"""

isa_data = pd.read_csv(str(Path(__file__).parent / "../isa_data/isa_sample_values.csv"))
altitudes = isa_data["Altitude [m]"].values
pressures = isa_data["Pressure [Pa]"].values
temperatures = isa_data["Temperature [K]"].values
densities = isa_data["Density [kg/m^3]"].values
speeds_of_sound = isa_data["Speed of Sound [m/s]"].values


def test_isa_atmosphere():
    for altitude, pressure, temperature, density, speed_of_sound in zip(
        altitudes, pressures, temperatures, densities, speeds_of_sound
    ):
        atmo = Atmosphere(altitude=altitude, method="isa")

        if (
            altitude >= atmo._valid_altitude_range[0]
            and altitude <= atmo._valid_altitude_range[1]
        ):
            fail_message = f"FAILED @ {altitude} m"

            assert atmo.pressure() == pytest.approx(pressure, abs=100), fail_message
            assert atmo.temperature() == pytest.approx(temperature, abs=1), fail_message
            assert atmo.density() == pytest.approx(density, abs=0.01), fail_message
            assert atmo.speed_of_sound() == pytest.approx(speed_of_sound, abs=1), (
                fail_message
            )


def test_isa_atmosphere_against_ussa1976_layer_bases():
    """
    Regression test: the ISA implementation should reproduce the official
    U.S. Standard Atmosphere 1976 (NASA-TM-X-74335) pressures at the layer
    base altitudes essentially exactly. This requires using the standard's
    g_0 = 9.80665 m/s^2 in the hydrostatic relation; a previous version used
    g = 9.81, which gave pressures up to ~0.4% low at high altitudes.

    Reference values are geopotential altitudes and pressures from the
    USSA1976 tables (which the ISA shares below 32 km, per ISO 2533).
    """
    ussa1976_pressures = {  # Geopotential altitude [m] -> Pressure [Pa]
        0: 101325.0,
        11000: 22632.06,
        20000: 5474.889,
        32000: 868.0187,
        47000: 110.9063,
        51000: 66.93887,
        71000: 3.956420,
    }
    for altitude, pressure in ussa1976_pressures.items():
        atmo = Atmosphere(altitude=altitude, method="isa")
        assert atmo.pressure() == pytest.approx(pressure, rel=1e-5), (
            f"FAILED @ {altitude} m"
        )


def test_isa_atmosphere_casadi_matches_numpy():
    """
    The ISA functions are dual-backend (NumPy + CasADi); both backends
    should produce identical numerics.
    """
    import casadi as cas
    from aerosandbox.atmosphere._isa_atmo_functions import (
        pressure_isa,
        temperature_isa,
    )

    altitudes_test = np.array([0.0, 5e3, 11e3, 32e3, 71e3])

    pressure_numpy = pressure_isa(altitudes_test)
    temperature_numpy = temperature_isa(altitudes_test)

    pressure_casadi = pressure_isa(cas.DM(altitudes_test)).full().flatten()
    temperature_casadi = temperature_isa(cas.DM(altitudes_test)).full().flatten()

    assert pressure_casadi == pytest.approx(pressure_numpy, rel=1e-12)
    assert temperature_casadi == pytest.approx(temperature_numpy, rel=1e-12)


def test_diff_atmosphere():
    altitudes = np.linspace(-50e2, 150e3, 1000)
    atmo_isa = Atmosphere(altitude=altitudes, method="isa")
    atmo_diff = Atmosphere(altitude=altitudes)
    temp_isa = atmo_isa.temperature()
    pressure_isa = atmo_isa.pressure()
    temp_diff = atmo_diff.temperature()
    pressure_diff = atmo_diff.pressure()
    assert max(abs((temp_isa - temp_diff) / temp_isa)) < 0.025, (
        "temperature failed for differentiable model"
    )
    assert max(abs((pressure_isa - pressure_diff) / pressure_isa)) < 0.01, (
        "pressure failed for differentiable model"
    )


def plot_isa_residuals():
    atmo = Atmosphere(altitude=altitudes, method="isa")

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


if __name__ == "__main__":
    # plot_isa_residuals()
    pytest.main()
