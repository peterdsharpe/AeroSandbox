import aerosandbox.numpy as np
import pandas as pd
from pathlib import Path

### Define constants
gas_constant_universal = 8.31432  # J/(mol*K); universal gas constant
molecular_mass_air = 28.9644e-3  # kg/mol; molecular mass of air
gas_constant_air = gas_constant_universal / molecular_mass_air  # J/(kg*K); gas constant of air
g = 9.81  # m/s^2, gravitational acceleration on earth

### Read ISA table data
isa_table = pd.read_csv(Path(__file__).parent.absolute() / "isa_data/isa_table.csv")
isa_base_altitude = isa_table["Base Altitude [m]"].values
isa_lapse_rate = isa_table["Lapse Rate [K/km]"].values / 1000
isa_base_temperature = isa_table["Base Temperature [C]"].values + 273.15


### Calculate pressure at each ISA level programmatically using the barometric pressure equation with linear temperature.
def barometric_formula(
        P_b,
        T_b,
        L_b,
        h,
        h_b,
):
    """
    The barometric pressure equation, from here: https://en.wikipedia.org/wiki/Barometric_formula
    Args:
        P_b: Pressure at the base of the layer, in Pa
        T_b: Temperature at the base of the layer, in K
        L_b: Temperature lapse rate, in K/m
        h: Altitude, in m
        h_b:

    Returns:

    """
    with np.errstate(divide="ignore", over="ignore"):
        T = T_b + L_b * (h - h_b)
        T = np.fmax(T, 0)  # Keep temperature nonnegative, no matter the inputs.
        if L_b != 0:
            return P_b * (T / T_b) ** (-g / (gas_constant_air * L_b))
        else:
            return P_b * np.exp(-g * (h - h_b) / (gas_constant_air * T_b))


isa_pressure = [101325.]  # Pascals
for i in range(len(isa_table) - 1):
    isa_pressure.append(
        barometric_formula(
            P_b=isa_pressure[i],
            T_b=isa_base_temperature[i],
            L_b=isa_lapse_rate[i],
            h=isa_base_altitude[i + 1],
            h_b=isa_base_altitude[i]
        )
    )


def pressure_isa(altitude):
    """
    Computes the pressure at a given altitude based on the International Standard Atmosphere.

    Uses the Barometric formula, as implemented here: https://en.wikipedia.org/wiki/Barometric_formula

    Args:
        altitude: Geopotential altitude [m]

    Returns: Pressure [Pa]

    """
    pressure = 0 * altitude  # Initialize the pressure to all zeros.

    for i in range(len(isa_table)):
        pressure = np.where(
            altitude > isa_base_altitude[i],
            barometric_formula(
                P_b=isa_pressure[i],
                T_b=isa_base_temperature[i],
                L_b=isa_lapse_rate[i],
                h=altitude,
                h_b=isa_base_altitude[i]
            ),
            pressure
        )

    ### Add lower bound case
    pressure = np.where(
        altitude <= isa_base_altitude[0],
        barometric_formula(
            P_b=isa_pressure[0],
            T_b=isa_base_temperature[0],
            L_b=isa_lapse_rate[0],
            h=altitude,
            h_b=isa_base_altitude[0]
        ),
        pressure
    )

    return pressure


def temperature_isa(altitude):
    """
    Computes the temperature at a given altitude based on the International Standard Atmosphere.

    Args:
        altitude: Geopotential altitude [m]

    Returns: Temperature [K]

    """
    temp = 0 * altitude  # Initialize the temperature to all zeros.

    for i in range(len(isa_table)):
        temp = np.where(
            altitude > isa_base_altitude[i],
            (altitude - isa_base_altitude[i]) * isa_lapse_rate[i] + isa_base_temperature[i],
            temp
        )

    ### Add lower bound case
    temp = np.where(
        altitude <= isa_base_altitude[0],
        (altitude - isa_base_altitude[0]) * isa_lapse_rate[0] + isa_base_temperature[0],
        temp
    )

    return temp


if __name__ == '__main__':
    pressure_isa(-50e3)
