import aerosandbox.numpy as np
from aerosandbox.common import AeroSandboxObject
from abc import ABC, abstractmethod, abstractproperty
from typing import Union, Dict, Tuple, List
from aerosandbox import MassProperties, Opti, OperatingPoint, Atmosphere
from aerosandbox.tools.string_formatting import trim_string


class _DynamicsPointMassBaseClass(AeroSandboxObject, ABC):
    @abstractmethod
    def __init__(self,
                 mass_props: MassProperties = None,
                 **state_variables_and_indirect_control_variables,
                 ):
        self.mass_props = MassProperties() if mass_props is None else mass_props
        """
        For each state variable, self.state_var = state_var 
        
        For each indirect control variable, self.indirect_control_var = indirect_control_var
        
        For each control variable, self.control_var = 0
        """

    @abstractproperty
    def state(self) -> Dict[str, Union[float, np.ndarray]]:
        pass

    def set_state(self,
                  value: Union[
                      Dict[str, Union[float, np.ndarray]],
                      List, Tuple, np.ndarray
                  ]
                  ):
        try:  # Assume `value` is a dict-like, with keys
            for key in self.state.keys():
                setattr(self, key, value[key])
        except TypeError:  # Assume it's an iterable that has been sorted.
            self.set_state(self.pack_state(value))

    def unpack_state(self,
                     dict_like_state: Dict[str, Union[float, np.ndarray]] = None
                     ) -> Tuple[Union[float, np.ndarray]]:
        """
        'Unpacks' a Dict-like state into an array-like that represents the state of the dynamical system.

        Args:
            dict_like_state: Takes in a dict-like representation of the state.

        Returns: The array representation of the state that you gave.

        """
        return tuple(self.state.values())

    def pack_state(self,
                   array_like_state: Union[List, Tuple, np.ndarray] = None
                   ) -> Dict[str, Union[float, np.ndarray]]:
        """
        'Packs' an array into a Dict that represents the state of the dynamical system.

        Args:
            array_like_state: Takes in an iterable that must have the same number of entries as the state vector of the system.

        Returns: The Dict representation of the state that you gave.

        """
        if array_like_state is None:
            return self.state
        if not len(self.state.keys()) == len(array_like_state):
            raise ValueError(
                "There are a differing number of elements in the `state` variable and the `array_like` you're trying to pack!")
        return {
            k: v
            for k, v in zip(
                self.state.keys(),
                array_like_state
            )
        }

    @abstractproperty
    def control_variables(self) -> Dict[str, Union[float, np.ndarray]]:
        pass

    def __repr__(self):

        title = f"{self.__class__.__name__} instance:"

        def makeline(k, v):
            name = trim_string(str(k).strip(), length=8).rjust(8)
            item = trim_string(str(v).strip(), length=40).ljust(40)

            line = f"{name}: {item}"

            return line

        state_variables_title = "\tState variables:"

        state_variables = "\n".join([
            "\t\t" + makeline(k, v)
            for k, v in self.state.items()
        ])

        control_variables_title = "\tControl variables:"

        control_variables = "\n".join([
            "\t\t" + makeline(k, v)
            for k, v in self.control_variables.items()
        ])

        return "\n".join([
            title,
            state_variables_title,
            state_variables,
            control_variables_title,
            control_variables
        ])

    def __getitem__(self, index):
        def get_item_of_attribute(a):
            try:
                return a[index]
            except TypeError:  # object is not subscriptable
                return a
            except IndexError as e:  # index out of range
                raise IndexError("A state variable could not be indexed, since the index is out of range!")

        state_variables = {
            k: get_item_of_attribute(v)
            for k, v in self.state.items()
        }

        mass_props = self.mass_props[index]

        return self.__class__(
            mass_props=mass_props,
            **state_variables  # TODO make this not break with indep. control vars.
        )

    @abstractmethod
    def state_derivatives(self) -> Dict[str, Union[float, np.ndarray]]:
        pass

    def constrain_derivatives(self,
                              opti: Opti = None,
                              time: np.ndarray = None,
                              which: Union[str, List[str]] = "all"
                              ):
        if which == "all":
            which = self.state.keys()

        state_derivatives = self.state_derivatives()

        for state_var_name in which:

            # If a state derivative has a None value, skip it.
            if state_derivatives[state_var_name] is None:
                continue

            # Try to constrain the derivative
            try:
                opti.constrain_derivative(
                    derivative=state_derivatives[state_var_name],
                    variable=self.state[state_var_name],
                    with_respect_to=time,
                )
            except KeyError:
                raise ValueError(f"This dynamics instance does not have a state named '{state_var_name}'!")
            except Exception as e:
                raise ValueError(f"Error while constraining state variable '{state_var_name}': \n{e}")

    @abstractmethod
    def convert_axes(self,
                     x_from: float,
                     y_from: float,
                     z_from: float,
                     from_axes: str,
                     to_axes: str,
                     ) -> Tuple[float, float, float]:
        """
        Converts a vector [x_from, y_from, z_from], as given in the `from_axes` frame, to an equivalent vector [x_to,
        y_to, z_to], as given in the `to_axes` frame.

        Identical to OperatingPoint.convert_axes(), but adds in "earth" as a valid axis frame. For more documentation,
        see the docstring of OperatingPoint.convert_axes().

        Both `from_axes` and `to_axes` should be a string, one of:
                * "geometry"
                * "body"
                * "wind"
                * "stability"
                * "earth"

        Args:
                x_from: x-component of the vector, in `from_axes` frame.
                y_from: y-component of the vector, in `from_axes` frame.
                z_from: z-component of the vector, in `from_axes` frame.
                from_axes: The axes to convert from.
                to_axes: The axes to convert to.

        Returns: The x-, y-, and z-components of the vector, in `to_axes` frame. Given as a tuple.

        """
        pass

    @abstractmethod
    def add_force(self,
                  Fx: Union[np.ndarray, float] = 0,
                  Fy: Union[np.ndarray, float] = 0,
                  Fz: Union[np.ndarray, float] = 0,
                  axes="wind",
                  ) -> None:
        pass

    def add_gravity_force(self,
                          g=9.81
                          ) -> None:
        self.add_force(
            Fz=self.mass_props.mass * g,
            axes="earth",
        )

    @property
    def op_point(self):
        return OperatingPoint(
            atmosphere=Atmosphere(altitude=self.altitude),
            velocity=self.speed,
            alpha=self.alpha,
            beta=self.beta,
            p=0,
            q=0,
            r=0,
        )

    @property
    def altitude(self):
        return -self.z_e

    @property
    def translational_kinetic_energy(self) -> float:
        return 0.5 * self.mass_props.mass * self.speed ** 2

    @property
    def kinetic_energy(self):
        return self.translational_kinetic_energy

    @property
    def potential_energy(self, g=9.81):
        """
        Gives the potential energy [J] from gravity.

        PE = mgh
        """
        return self.mass_props.mass * g * self.altitude
