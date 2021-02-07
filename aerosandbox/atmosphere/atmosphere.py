from aerosandbox.common import AeroSandboxObject
import aerosandbox.numpy as np
import pandas as pd
from pathlib import Path

### Define constants
gas_constant_universal = 8.31432  # J/(mol*K); universal gas constant
molecular_mass_air = 28.9644e-3  # kg/mol; molecular mass of air
gas_constant_air = gas_constant_universal / molecular_mass_air  # J/(kg*K); gas constant of air
g = 9.81  # m/s^2, gravitational acceleration on earth
effective_collision_diameter = 0.365e-9 # m, effective collision diameter of an air molecule
ratio_of_specific_heats = 1.4 # unitless, ratio of specific heats of air

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
                * "differentiable" - a C1-continuous fit to the International Standard Atmosphere
                * "isa" - the International Standard Atmosphere
                
        """
        self.altitude = altitude
        self.type = type
        self._valid_altitude_range = (0, 80000)

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
        """
        Computes the pressure at the Atmosphere's altitude based on the International Standard Atmosphere.

        Uses the Barometric formula, as implemented here: https://en.wikipedia.org/wiki/Barometric_formula

        Returns: Pressure [Pa]

        """
        alt = self.altitude
        pressure = 0 * alt  # Initialize the pressure to all zeros.

        for i in range(len(isa_table)):
            pressure = np.where(
                alt > isa_base_altitude[i],
                barometric_formula(
                    P_b=isa_pressure[i],
                    T_b=isa_base_temperature[i],
                    L_b=isa_lapse_rate[i],
                    h=alt,
                    h_b=isa_base_altitude[i]
                ),
                pressure
            )

        ### Add lower bound case
        pressure = np.where(
            alt <= isa_base_altitude[0],
            barometric_formula(
                P_b=isa_pressure[0],
                T_b=isa_base_temperature[0],
                L_b=isa_lapse_rate[0],
                h=alt,
                h_b=isa_base_altitude[0]
            ),
            pressure
        )

        return pressure

    def _temperature_isa(self):
        """
        Computes the temperature at the Atmosphere's altitude based on the International Standard Atmosphere.
        Returns: Temperature [K]

        """
        alt = self.altitude
        temp = 0 * alt  # Initialize the temperature to all zeros.

        for i in range(len(isa_table)):
            temp = np.where(
                alt > isa_base_altitude[i],
                (alt - isa_base_altitude[i]) * isa_lapse_rate[i] + isa_base_temperature[i],
                temp
            )

        ### Add lower bound case
        temp = np.where(
            alt <= isa_base_altitude[0],
            (alt - isa_base_altitude[0]) * isa_lapse_rate[0] + isa_base_temperature[0],
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
    altitude = np.linspace(-5000, 100000, 500)
    atmo_diff = Atmosphere(altitude=altitude)
    atmo_isa = Atmosphere(altitude=altitude, type="isa")

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(palette=sns.color_palette("husl"))
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)


    plt.semilogx(
        atmo_isa.pressure(),
        atmo_isa.altitude/1e3,
        label="ISA Ref."
    )
    lims = ax.get_xlim(), ax.get_ylim()
    plt.semilogx(
        atmo_diff.pressure(),
        atmo_diff.altitude/1e3,
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
        atmo_isa.altitude/1e3,
        label="ISA Ref.",
    )
    lims = ax.get_xlim(), ax.get_ylim()
    plt.plot(
        atmo_diff.temperature(),
        atmo_diff.altitude/1e3,
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

