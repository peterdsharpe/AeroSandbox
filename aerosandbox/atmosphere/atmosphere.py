from aerosandbox.common import AeroSandboxObject
import aerosandbox.numpy as np
import pandas as pd
from pathlib import Path
from aerosandbox.modeling.interpolation import InterpolatedModel
from aerosandbox.atmosphere._isa_atmo_functions import pressure_isa, temperature_isa
from aerosandbox.atmosphere._diff_atmo_functions import pressure_differentiable, temperature_differentiable

### Define constants
gas_constant_universal = 8.31432  # J/(mol*K); universal gas constant
molecular_mass_air = 28.9644e-3  # kg/mol; molecular mass of air
gas_constant_air = gas_constant_universal / molecular_mass_air  # J/(kg*K); gas constant of air
g = 9.81  # m/s^2, gravitational acceleration on earth
effective_collision_diameter = 0.365e-9  # m, effective collision diameter of an air molecule
ratio_of_specific_heats = 1.4  # unitless, ratio of specific heats of air

### Define the Atmosphere class
class Atmosphere(AeroSandboxObject):
    r"""
    All models here are smoothed fits to the 1976 COESA model;
    see AeroSandbox\studies\Atmosphere Fitting for details.

    All models considered valid from 0 to 40 km.
    """

    def __init__(self,
                 altitude: float = 0.,  # meters
                 method: str = "differentiable"
                 ):
        """
        Initialize a new Atmosphere.
        
        Args:
            
            altitude: Flight altitude, in meters. This is assumed to be a geopotential altitude above MSL.
            
            method: Method of atmosphere modeling to use. Either:
                * "differentiable" - a C1-continuous fit to the International Standard Atmosphere
                * "isa" - the International Standard Atmosphere
                
        """
        self.altitude = altitude
        self.method = method
        self._valid_altitude_range = (0, 80000)

    ### The two primary state variables, pressure and temperature, go here!

    def pressure(self):
        """
        Returns the pressure, in Pascals.
        """
        if self.method.lower() == "isa":
            return pressure_isa(self.altitude)
        elif self.method.lower() == "differentiable":
            return pressure_differentiable(self.altitude)
        else:
            raise ValueError("Bad value of 'type'!")

    def temperature(self):
        """
        Returns the temperature, in Kelvin.
        """
        if self.method.lower() == "isa":
            return temperature_isa(self.altitude)
        elif self.method.lower() == "differentiable":
            return temperature_differentiable(self.altitude)
        else:
            raise ValueError("Bad value of 'type'!")

    ### Everything else in this class is a derived quantity; all models of derived quantities go here.

    def density(self):
        """
        Returns the density, in kg/m^3.
        """
        rho = self.pressure() / (self.temperature() * gas_constant_air)

        return rho

    def speed_of_sound(self):
        """
        Returns the speed of sound, in m/s.
        """
        temperature = self.temperature()
        return (ratio_of_specific_heats * gas_constant_air * temperature) ** 0.5

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
        temperature = self.temperature()
        mu = C1 * temperature ** 1.5 / (temperature + S)

        return mu

    # def thermal_velocity(self):
    #     """
    #     Returns the thermal velocity (mean particle speed)
    #     Returns:
    #
    #     """
    #
    # def mean_free_path(self): # TODO finish implementing methods for our hypersonics friends
    #     """Returns the mean free path of an air molecule, in meters."""
    #     return 1/(
    #         2 ** 0.5 * np.pi *
    #     )


if __name__ == "__main__":
    # Make AeroSandbox Atmosphere
    altitude = np.linspace(-5e3, 100e3, 1000)
    atmo_diff = Atmosphere(altitude=altitude)
    atmo_isa = Atmosphere(altitude=altitude, method="isa")

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(palette=sns.color_palette("husl", 2))
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)

    plt.semilogx(
        atmo_isa.pressure(),
        atmo_isa.altitude / 1e3,
        label="ISA Ref."
    )
    lims = ax.get_xlim(), ax.get_ylim()
    plt.semilogx(
        atmo_diff.pressure(),
        atmo_diff.altitude / 1e3,
        label="ASB Atmo."
    )
    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[1])
    plt.xlabel("Pressure [Pa]")
    plt.ylabel("Altitude [km]")
    plt.title("AeroSandbox Atmosphere vs. ISA Atmosphere")
    plt.legend()
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
    plt.plot(
        atmo_isa.temperature(),
        atmo_isa.altitude / 1e3,
        label="ISA Ref.",
    )
    lims = ax.get_xlim(), ax.get_ylim()
    plt.plot(
        atmo_diff.temperature(),
        atmo_diff.altitude / 1e3,
        label="ASB Atmo."
    )
    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[1])
    plt.xlabel("Temperature [K]")
    plt.ylabel("Altitude [km]")
    plt.title("AeroSandbox Atmosphere vs. ISA Atmosphere")
    plt.legend()
    plt.tight_layout()
    plt.show()
