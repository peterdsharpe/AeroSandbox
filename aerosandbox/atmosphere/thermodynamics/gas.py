import aerosandbox.numpy as np
from aerosandbox.tools.string_formatting import eng_string
import copy

universal_gas_constant = 8.31432  # J/(mol*K); universal gas constant


# def ideal_gas_law(
#
#         molecular_mass = 28.9644e-3,
# )

class IdealPerfectGas:
    def __init__(self,
                 pressure=101325,
                 temperature=273.15 + 15,
                 specific_heat_constant_pressure=1006,
                 specific_heat_constant_volume=717,
                 molecular_mass=28.9644e-3,
                 effective_collision_diameter=0.365e-9,
                 ):
        """

        Args:

            pressure: Pressure of the gas, in Pascals

            temperature: Temperature of the gas, in Kelvin

            specific_heat_constant_pressure: Specific heat at constant pressure, also known as C_p. In J/kg-K.

            specific_heat_constant_volume: Specific heat at constant volume, also known as C_v. In J/kg-K.

            molecular_mass: Molecular mass of the gas, in kg/mol

            effective_collision_diameter: Effective collision diameter of a molecule, in meters.

        """
        self.pressure = pressure
        self.temperature = temperature
        self.specific_heat_constant_pressure = specific_heat_constant_pressure
        self.specific_heat_constant_volume = specific_heat_constant_volume
        self.molecular_mass = molecular_mass
        self.effective_collision_diameter = effective_collision_diameter

    def __repr__(self) -> str:
        f = lambda s, u: eng_string(s, unit=u, format="%.6g")

        return f"Gas (P = {f(self.pressure, 'Pa')}, T = {self.temperature:.6g} K, ρ = {self.density:.6g} kg/m^3, Pv^gamma = {self.pressure * self.specific_volume ** self.ratio_of_specific_heats: .6g})"

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        return self.__dict__ == other.__dict__

    @property
    def density(self):
        return self.pressure / (self.temperature * self.specific_gas_constant)

    @property
    def speed_of_sound(self):
        return (self.ratio_of_specific_heats * self.specific_gas_constant * self.temperature) ** 0.5

    @property
    def specific_gas_constant(self):
        return universal_gas_constant / self.molecular_mass

    @property
    def ratio_of_specific_heats(self):
        return self.specific_heat_constant_pressure / self.specific_heat_constant_volume

    @property
    def specific_volume(self):
        return 1 / self.density

    def process(self,
                process: str = "isentropic",
                new_pressure: float = None,
                new_temperature: float = None,
                new_density: float = None,
                polytropic_n: float = None,
                inplace=False
                ):
        """
        Puts this gas under a thermodynamic process.

        Equations here: https://en.wikipedia.org/wiki/Ideal_gas_law

        Args:

            process: Type of process. One of:

                * "isobaric"
                * "isochoric"
                * "isothermal"
                * "isentropic"
                * "polytropic"

            new_pressure: The pressure that the

            new_temperature:

            new_density:

            polytropic_n:

            inplace:


        Returns:

        """
        pressure_specified = new_pressure is not None
        temperature_specified = new_temperature is not None
        density_specified = new_density is not None

        number_of_conditions_specified = (pressure_specified + temperature_specified + density_specified)

        if number_of_conditions_specified == 0:
            raise ValueError("You must specify a new pressure, temperature, or density for this process to go to.")
        elif number_of_conditions_specified > 1:
            raise ValueError(
                "You can only specify only one of pressure, temperature or density; the state is overdetermined otherwise.")

        if pressure_specified:
            P_ratio = new_pressure / self.pressure
        elif temperature_specified:
            T_ratio = new_temperature / self.temperature
        elif density_specified:
            V_ratio = 1 / (new_density / self.density)

        if process == "isobaric":

            new_pressure = self.pressure

            if pressure_specified:
                raise ValueError("Can't specify pressure change for an isobaric process!")

            elif density_specified:
                new_temperature = self.temperature * V_ratio

            elif temperature_specified:
                pass

        elif process == "isochoric":

            if pressure_specified:
                new_temperature = self.temperature * P_ratio

            elif density_specified:
                raise ValueError("Can't specify density change for an isochoric process!")

            elif temperature_specified:
                new_pressure = self.pressure * T_ratio

        elif process == "isothermal":

            new_temperature = self.temperature

            if pressure_specified:
                pass

            elif density_specified:
                new_pressure = self.pressure / V_ratio

            elif temperature_specified:
                raise ValueError("Can't specify temperature change for an isothermal process!")

        elif process == "isentropic":

            gam = self.ratio_of_specific_heats

            if pressure_specified:
                new_temperature = self.temperature * P_ratio ** ((gam - 1) / gam)

            elif density_specified:
                new_pressure = self.pressure * V_ratio ** -gam
                new_temperature = self.temperature * V_ratio ** (1 - gam)

            elif temperature_specified:
                new_pressure = self.pressure * T_ratio ** (gam / (gam - 1))

        elif process == "polytropic":

            if polytropic_n is None:
                raise ValueError("If the process is polytropic, then the polytropic index `n` must be specified.")

            n = polytropic_n

            if pressure_specified:
                new_temperature = self.temperature * P_ratio ** ((n - 1) / n)

            elif density_specified:
                new_pressure = self.pressure * V_ratio ** -n
                new_temperature = self.temperature * V_ratio ** (1 - n)

            elif temperature_specified:
                new_pressure = self.pressure * T_ratio ** (n / (n - 1))

        elif process == "isenthalpic":
            raise NotImplementedError()

        else:
            raise ValueError("Bad value of `process`!")

        if inplace:
            self.pressure = new_pressure
            self.temperature = new_temperature

        else:
            return IdealPerfectGas(
                pressure=new_pressure,
                temperature=new_temperature,
                specific_heat_constant_pressure=self.specific_heat_constant_pressure,
                specific_heat_constant_volume=self.specific_heat_constant_volume,
                molecular_mass=self.molecular_mass,
                effective_collision_diameter=self.effective_collision_diameter
            )


if __name__ == '__main__':
    air = IdealPerfectGas()
    # print(air)

    # air = air.process("isentropic", new_pressure=1e6)
    # print(air)

    ### Carnot
    print(air)
    air = air.process("isothermal", new_density=5)
    print(air)
    air = air.process("isentropic", new_density=0.1)
    print(air)
    air = air.process("isothermal", new_density=5)
    print(air)
    air = air.process("isentropic", new_pressure=IdealPerfectGas().pressure)
    print(air)
