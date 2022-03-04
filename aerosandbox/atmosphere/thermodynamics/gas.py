import aerosandbox.numpy as np
from aerosandbox.tools.string_formatting import eng_string
import copy
from typing import Union

universal_gas_constant = 8.31432  # J/(mol*K); universal gas constant


class PerfectGas:
    """
    Provides a class for an ideal, calorically perfect gas.

    Specifically, this gas:
        * Has PV = nRT (ideal)
        * Has constant heat capacities C_V, C_P (independent of temperature and pressure).
        * Is in thermodynamic equilibrium
        * Is not chemically reacting
        * Has internal energy and enthalpy purely as functions of temperature
    """

    def __init__(self,
                 pressure: Union[float, np.ndarray] = 101325,
                 temperature: Union[float, np.ndarray] = 273.15 + 15,
                 specific_heat_constant_pressure: float = 1006,
                 specific_heat_constant_volume: float = 717,
                 molecular_mass: float = 28.9644e-3,
                 effective_collision_diameter: float = 0.365e-9,
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

    def specific_enthalpy_change(self, start_temperature, end_temperature):
        """
        Returns the change in specific enthalpy that would occur from a given temperature change via a thermodynamic
        process.

        Args:
            start_temperature: Starting temperature [K]
            end_temperature: Ending temperature [K]

        Returns: The change in specific enthalpy, in J/kg.

        """
        return self.specific_heat_constant_pressure * (end_temperature - start_temperature)

    def specific_internal_energy_change(self, start_temperature, end_temperature):
        """
        Returns the change in specific internal energy that would occur from a given temperature change via a
        thermodynamic process.

        Args:
            start_temperature: Starting temperature [K]
            end_temperature: Ending temperature [K]

        Returns: The change in specific internal energy, in J/kg.

        """
        return self.specific_heat_constant_volume * (end_temperature - start_temperature)

    @property
    def specific_volume(self):
        """
        Gives the specific volume, often denoted `v`.

        (Note the lowercase; "V" is often the volume of a specific amount of gas, and this presents a potential point
        of confusion.)
        """
        return 1 / self.density

    @property
    def specific_enthalpy(self):
        """
        Gives the specific enthalpy, often denoted `h`.

        Enthalpy here is in units of J/kg.
        """
        return self.specific_enthalpy_change(start_temperature=0, end_temperature=self.temperature)

    @property
    def specific_internal_energy(self):
        """
        Gives the specific internal energy, often denoted `u`.

        Internal energy here is in units of J/kg.
        """
        return self.specific_internal_energy_change(start_temperature=0, end_temperature=self.temperature)

    def process(self,
                process: str = "isentropic",
                new_pressure: float = None,
                new_temperature: float = None,
                new_density: float = None,
                enthalpy_addition_at_constant_pressure: float = None,
                enthalpy_addition_at_constant_volume: float = None,
                polytropic_n: float = None,
                inplace=False
                ) -> "PerfectGas":
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

                The `process` must be specified.

            You must specifiy exactly one of the following arguments:
                * `new_pressure`: the new pressure after the process [Pa].
                * `new_temperature`: the new temperature after the process [K]
                * `new_density`: the new density after the process [kg/m^3]
                * `enthalpy_addition_at_constant_pressure`: [J/kg]
                * `enthalpy_addition_at_constant_volume`: [J/kg]

            polytropic_n: If you specified the process type to be "polytropic", you must provide the polytropic index
            `n` to be used here. (Reminder: PV^n = constant)

            inplace: Specifies whether to return the result in-place or to allocate a new PerfectGas object in memory
            for the result.

        Returns:

            If `inplace` is False (default), returns a new PerfectGas object that represents the gas after the change.

            If `inplace` is True, nothing is returned.

        """

        pressure_specified = new_pressure is not None
        temperature_specified = new_temperature is not None
        density_specified = new_density is not None
        enthalpy_at_pressure_specified = enthalpy_addition_at_constant_pressure is not None
        enthalpy_at_volume_specified = enthalpy_addition_at_constant_volume is not None

        number_of_conditions_specified = (
                pressure_specified +
                temperature_specified +
                density_specified +
                enthalpy_at_pressure_specified +
                enthalpy_at_volume_specified
        )

        if number_of_conditions_specified != 1:
            raise ValueError("You must specify exactly one of the following arguments:\n" + "\n".join([
                "\t* `new_pressure`",
                "\t* `new_temperature`",
                "\t* `new_density`",
                "\t* `enthalpy_addition_at_constant_pressure`",
                "\t* `enthalpy_addition_at_constant_volume`",
            ]))

        if enthalpy_at_pressure_specified:
            new_temperature = self.temperature + enthalpy_addition_at_constant_pressure / self.specific_heat_constant_pressure
        elif enthalpy_at_volume_specified:
            new_temperature = self.temperature + enthalpy_addition_at_constant_volume / self.specific_heat_constant_volume

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
            return PerfectGas(
                pressure=new_pressure,
                temperature=new_temperature,
                specific_heat_constant_pressure=self.specific_heat_constant_pressure,
                specific_heat_constant_volume=self.specific_heat_constant_volume,
                molecular_mass=self.molecular_mass,
                effective_collision_diameter=self.effective_collision_diameter
            )


if __name__ == '__main__':

    ### Carnot
    g = []

    g.append(PerfectGas(pressure=100e3, temperature=300))
    g.append(g[-1].process("isothermal", new_density=0.5))
    g.append(g[-1].process("isentropic", new_density=0.25))
    g.append(g[-1].process("isothermal", new_density=0.58))
    g.append(g[-1].process("isentropic", new_temperature=300))

    for i in range(len(g)):
        print(f"After Process {i}: {g[i]}")
