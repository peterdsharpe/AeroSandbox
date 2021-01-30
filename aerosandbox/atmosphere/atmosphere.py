from aerosandbox.common import AeroSandboxObject
import numpy as np

R_universal = 8.31432  # J/(mol*K); universal gas constant
M_air = 28.9644e-3  # kg/mol; molecular mass of air
R_air = R_universal / M_air  # J/(kg*K); gas constant of air


class Atmosphere(AeroSandboxObject):
    r"""
    All models here are smoothed fits to the 1976 COESA model;
    see AeroSandbox\studies\Atmosphere Fitting for details.

    All models considered valid from 0 to 40 km.
    """

    def __init__(self,
                 altitude: float = 0.  # meters
                 ):
        self.altitude = altitude

    def pressure(self):
        """
        Returns the pressure, in Pascals.
        """
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

    def temperature(self):
        """
        Returns the temperature, in Kelvin.
        """

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
    test_altitudes = cas.linspace(0, 40000, 201)
    test_pressures = get_pressure_at_altitude(test_altitudes)
    test_temps = get_temperature_at_altitude(test_altitudes)
    import matplotlib.pyplot as plt
    import matplotlib.style as style
    import plotly.express as px
    import plotly.graph_objects as go

    style.use("seaborn")

    plt.semilogy(test_altitudes, test_pressures)
    plt.xlabel("Altitude [m]")
    plt.ylabel("Pressure [Pa]")
    plt.show()

    plt.plot(test_altitudes, test_temps)
    plt.xlabel("Altitude [m]")
    plt.ylabel("Temperature [K]")
    plt.show()
