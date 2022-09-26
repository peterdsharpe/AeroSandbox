from aerosandbox.common import AeroSandboxObject
import aerosandbox.numpy as np
from aerosandbox.atmosphere._isa_atmo_functions import pressure_isa, temperature_isa
from aerosandbox.atmosphere._diff_atmo_functions import pressure_differentiable, temperature_differentiable

### Define constants
gas_constant_universal = 8.31432  # J/(mol*K); universal gas constant
molecular_mass_air = 28.9644e-3  # kg/mol; molecular mass of air
gas_constant_air = gas_constant_universal / molecular_mass_air  # J/(kg*K); gas constant of air
effective_collision_diameter = 0.365e-9  # m, effective collision diameter of an air molecule


### Define the Atmosphere class
class Atmosphere(AeroSandboxObject):
    r"""
    All models here are smoothed fits to the 1976 COESA model;
    see AeroSandbox\studies\Atmosphere Fitting for details.

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
        return (self.ratio_of_specific_heats() * gas_constant_air * temperature) ** 0.5

    def dynamic_viscosity(self):
        """
        Returns the dynamic viscosity (mu), in kg/(m*s).

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

    def kinematic_viscosity(self):
        """
        Returns the kinematic viscosity (nu), in m^2/s.

        Definitional.
        """
        return self.dynamic_viscosity() / self.density()

    def ratio_of_specific_heats(self):
        return 1.4  # TODO model temperature variation

    # def thermal_velocity(self):
    #     """
    #     Returns the thermal velocity (mean particle speed)
    #     Returns:
    #
    #     """
    #
    def mean_free_path(self):
        """
        Returns the mean free path of an air molecule, in meters.

        To find the collision radius, assumes "a hard-sphere gas that has the same viscosity as the actual gas being considered".

        From Vincenti, W. G. and Kruger, C. H. (1965). Introduction to physical gas dynamics. Krieger Publishing Company. p. 414.

        """
        return self.dynamic_viscosity() / self.pressure() * np.sqrt(
            np.pi * gas_constant_universal * self.temperature() / (2 * molecular_mass_air)
        )

    def knudsen(self, length):
        """
        Computes the Knudsen number for a given length.
        """
        return self.mean_free_path() / length


if __name__ == "__main__":
    # Make AeroSandbox Atmosphere
    altitude = np.linspace(-5e3, 100e3, 1000)
    atmo_diff = Atmosphere(altitude=altitude)
    atmo_isa = Atmosphere(altitude=altitude, method="isa")

    from aerosandbox.tools.pretty_plots import plt, sns, mpl, show_plot, set_ticks

    fig, ax = plt.subplots()

    plt.plot(
        (
                (atmo_diff.pressure() - atmo_isa.pressure()) / atmo_isa.pressure()
        ) * 100,
        altitude / 1e3,
    )
    set_ticks(0.2, 0.1, 20, 10)
    plt.xlim(-1, 1)
    show_plot(
        "AeroSandbox Atmosphere vs. ISA Atmosphere",
        "Pressure, Relative Error [%]",
        "Altitude [km]"
    )

    fig, ax = plt.subplots()
    plt.plot(
        atmo_diff.temperature() - atmo_isa.temperature(),
        altitude / 1e3,
    )
    set_ticks(1, 0.5, 20, 10)
    plt.xlim(-5, 5)
    show_plot(
        "AeroSandbox Atmosphere vs. ISA Atmosphere",
        "Temperature, Absolute Error [K]",
        "Altitude [km]"
    )
