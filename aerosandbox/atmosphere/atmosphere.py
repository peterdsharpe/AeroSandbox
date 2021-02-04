from aerosandbox.common import AeroSandboxObject
import numpy as np
from aerosandbox import linspace, if_else
import pandas as pd
from pathlib import Path

R_universal = 8.31432  # J/(mol*K); universal gas constant
M_air = 28.9644e-3  # kg/mol; molecular mass of air
R_air = R_universal / M_air  # J/(kg*K); gas constant of air

### Read ISA table data
isa_table = pd.read_csv(Path(__file__).parent.absolute() / "isa_data/isa_table.csv")
isa_base_altitude = isa_table["Base Altitude [m]"].values
isa_lapse_rate = isa_table["Lapse Rate [K/km]"].values / 1000
isa_base_temperature = isa_table["Base Temperature [C]"].values + 273.15


### Define the Atmosphere class
class Atmosphere(AeroSandboxObject):
    r"""
    All models here are smoothed fits to the 1976 COESA model;
    see AeroSandbox\studies\Atmosphere Fitting for details.

    All models considered valid from 0 to 40 km.
    """

    def __init__(self,
                 altitude: float = 0.,  # meters
                 type: str = "differentiable"
                 ):
        """
        Initialize a new Atmosphere.
        
        Args:
            
            altitude: Flight altitude, in meters. This is assumed to be a geopotential altitude above MSL.
            
            type: Type of atmosphere that you want. Either:
                * "differentiable" - a fit to the International Standard Atmosphere
                * "isa" - the International Standard Atmosphere
                
        """
        self.altitude = altitude
        self.type = type
        self._valid_altitude_range = (0, 40000)

    ### The two primary state variables, pressure and temperature, go here!

    def pressure(self):
        """
        Returns the pressure, in Pascals.
        """
        if self.type == "isa":
            return self._pressure_isa()
        elif self.type == "differentiable":
            return self._pressure_differentiable()
        else:
            raise ValueError("Bad value of 'type'!")

    def temperature(self):
        """
        Returns the temperature, in Kelvin.
        """
        if self.type == "isa":
            return self._temperature_isa()
        elif self.type == "differentiable":
            return self._temperature_differentiable()
        else:
            raise ValueError("Bad value of 'type'!")

    ### Individual models for the two primary state variables, pressure and temperature, go here!

    def _pressure_isa(self):
        return 5e4 * np.ones_like(self.altitude)

    def _temperature_isa(self):
        """
        Computes the temperature at the Atmosphere's altitude based on the International Standard Atmosphere.
        Returns:

        """
        alt = self.altitude
        temp = 0 * alt  # Initialize the temperature to all zeros.

        for i in range(len(isa_table)):
            temp = if_else(
                alt > isa_base_altitude[i],
                (alt - isa_base_altitude[i]) * isa_lapse_rate[i] + isa_base_temperature[i],
                temp
            )

        ### Add lower bound case
        temp = if_else(
            alt < isa_base_altitude[0],
            isa_base_temperature[0],
            temp
        )

        return temp

    # return 260 * np.ones_like(self.altitude)

    def _pressure_differentiable(self):
        altitude_scaled = self.altitude / 40000

        p1 = -1.822942e+00
        p2 = 5.366751e+00
        p3 = -5.021452e+00
        p4 = -4.424532e+00
        p5 = 1.151986e+01

        x = altitude_scaled
        logP = p5 + x * (p4 + x * (p3 + x * (p2 + x * (p1))))

        pressure = np.exp(logP)

        return pressure

    def _temperature_differentiable(self):

        altitude_scaled = self.altitude / 40000

        p1 = -2.122102e+01
        p2 = 7.000812e+01
        p3 = -8.759170e+01
        p4 = 5.047893e+01
        p5 = -1.176537e+01
        p6 = -3.566535e-02
        p7 = 5.649588e+00

        x = altitude_scaled
        logT = p7 + x * (p6 + x * (p5 + x * (p4 + x * (p3 + x * (p2 + x * (p1))))))

        temperature = np.exp(logT)

        return temperature

    ### Everything else in this class is a derived quantity; all models of derived quantities go here.

    def density(self):
        """
        Returns the density, in kg/m^3.
        """
        rho = self.pressure() / (self.temperature() * R_air)

        return rho

    def speed_of_sound(self):
        """
        Returns the speed of sound, in m/s.
        """
        temperature = self.temperature()
        return (1.4 * R_air * temperature) ** 0.5

    def dynamic_viscosity(self):
        """
        Returns the dynamic viscosity, in kg/(m*s).

        Based on Sutherland's Law, citing `https://www.cfd-online.com/Wiki/Sutherland's_law`.

        According to Rathakrishnan, E. (2013). Theoretical aerodynamics. John Wiley & Sons.:
        This relationship is valid from 0.01 to 100 atm, and between 0 and 3000K.

        According to White, F. M., & Corfield, I. (2006). Viscous fluid flow (Vol. 3, pp. 433-434). New York: McGraw-Hill.:
        The error is no more than approximately 2% for air between 170K and 1900K.
        """

        # Sutherland constants
        C1 = 1.458e-6  # kg/(m*s*sqrt(K))
        S = 110.4  # K

        # Sutherland equation
        mu = C1 * self.temperature() ** 1.5 / (self.temperature() + S)

        return mu


if __name__ == "__main__":
    # Make AeroSandbox Atmosphere
    altitude = linspace(-5000, 110000, 201)
    atmo_diff = Atmosphere(altitude=altitude)
    atmo_isa = Atmosphere(altitude=altitude, type="isa")

    import matplotlib.pyplot as plt
    import matplotlib.style as style

    style.use("seaborn")

    plt.semilogy(
        atmo_diff.altitude,
        atmo_diff.pressure(),
        label="ASB Atmo."
    )
    plt.semilogy(
        atmo_isa.altitude,
        atmo_isa.pressure(),
        label="ISA Ref."
    )
    plt.xlabel("Altitude [m]")
    plt.ylabel("Pressure [Pa]")
    plt.title("AeroSandbox Atmosphere vs. ISA Atmosphere")
    plt.legend()
    plt.show()

    plt.plot(
        atmo_diff.altitude,
        atmo_diff.temperature(),
        label="ASB Atmo."
    )
    plt.plot(
        atmo_isa.altitude,
        atmo_isa.temperature(),
        label="ISA Ref.",
    )
    plt.xlabel("Altitude [m]")
    plt.ylabel("Temperature [K]")
    plt.title("AeroSandbox Atmosphere vs. ISA Atmosphere")
    plt.legend()
    plt.show()
