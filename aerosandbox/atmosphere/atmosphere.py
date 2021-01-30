from aerosandbox.common import AeroSandboxObject
import numpy as np

R_universal = 8.31432  # J/(mol*K); universal gas constant
M_air = 28.9644e-3  # kg/mol; molecular mass_total of air
R_air = R_universal / M_air  # Gas constant of air

class Atmosphere(AeroSandboxObject):
    def __init__(self,
                 altitude: float=0. # meters
                 ):
        self.altitude = altitude

    def pressure(self):
        r"""
        Smoothed fit to the 1976 COESA model; see AeroSandbox\studies\Atmosphere Fitting for details.
        Valid from 0 to 40 km.
        :param altitude: Altitude [meters]
        :return: Pressure [Pa]
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
        r"""
        Smoothed fit to the 1976 COESA model; see AeroSandbox\studies\Atmosphere Fitting for details.
        Valid from 0 to 40 km.
        :param altitude: Altitude [meters]
        :return: Temperature [K]
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
        Returns the density as a function of altitude. Based on the 1976 COESA model.
        There's a more efficient to do this using equation of state, but you can use this if you really want.
        Args:
            altitude: Altitude [meters]

        Returns: Density [kg/m^3]
        """

        P = get_pressure_at_altitude(altitude)
        T = get_temperature_at_altitude(altitude)

        rho = P / (T * R_air)

        return rho

    def speed_of_sound(self):
        """
        Finds the speed of sound from a specified temperature.
        Assumes ideal gas properties and ratio of specific heats of 1.4.
        :param temperature: Temperature [K]
        :return: Speed of sound [m/s]
        """
        temperature = self.temperature()
        return (1.4 * R_air * temperature) ** 0.5

    def dynamic_viscosity(self):
        """
        Finds the dynamic viscosity of air from a specified temperature. Uses Sutherland's Law
        :param temperature: Temperature, in Kelvin
        :return: Dynamic viscosity, in kg/(m*s)
        """
        # Uses Sutherland's law from the temperature
        # From `https://www.cfd-online.com/Wiki/Sutherland's_law`

        temperature = self.temperature()

        # Sutherland constant
        C1 = 1.458e-6  # kg/(m*s*sqrt(K))
        S = 110.4  # K

        mu = C1 * temperature ** (3 / 2) / (temperature + S)
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