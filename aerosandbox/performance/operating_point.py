from aerosandbox.common import AeroSandboxObject
from aerosandbox import Atmosphere
import aerosandbox.numpy as np
from typing import Tuple, Union, Dict, List
from aerosandbox.tools.string_formatting import trim_string
import inspect


class OperatingPoint(AeroSandboxObject):
    def __init__(self,
                 atmosphere: Atmosphere = Atmosphere(altitude=0),
                 velocity: float = 1.,
                 alpha: float = 0.,
                 beta: float = 0.,
                 p: float = 0.,
                 q: float = 0.,
                 r: float = 0.,
                 ):
        """
        An object that represents the instantaneous aerodynamic flight conditions of an aircraft.

        Args:
            atmosphere: The atmosphere object (of type asb.Atmosphere). Defaults to sea level conditions.

            velocity: The flight velocity, expressed as a true airspeed. [m/s]

            alpha: The angle of attack. [degrees]

            beta: The sideslip angle. (Reminder: convention that a positive beta implies that the oncoming air comes
            from the pilot's right-hand side.) [degrees]

            p: The roll rate about the x_b axis. [rad/sec]

            q: The pitch rate about the y_b axis. [rad/sec]

            r: The yaw rate about the z_b axis. [rad/sec]

        """
        self.atmosphere = atmosphere
        self.velocity = velocity
        self.alpha = alpha
        self.beta = beta
        self.p = p
        self.q = q
        self.r = r

    @property
    def state(self) -> Dict[str, Union[float, np.ndarray]]:
        """
        Returns the state variables of this OperatingPoint instance as a Dict.

        Keys are strings that give the name of the variables.
        Values are the variables themselves.

        """
        return {
            "atmosphere": self.atmosphere,
            "velocity"  : self.velocity,
            "alpha"     : self.alpha,
            "beta"      : self.beta,
            "p"         : self.p,
            "q"         : self.q,
            "r"         : self.r,
        }

    def get_new_instance_with_state(self,
                                    new_state: Union[
                                        Dict[str, Union[float, np.ndarray]],
                                        List, Tuple, np.ndarray
                                    ] = None
                                    ):
        """
        Creates a new instance of the OperatingPoint class from the given state.

        Args:
            new_state: The new state to be used for the new instance. Ideally, this is represented as a Dict in identical format to the `state` of a OperatingPoint instance.

        Returns: A new instance of this same OperatingPoint class.

        """

        ### Get a list of all the inputs that the class constructor wants to see
        init_signature = inspect.signature(self.__class__.__init__)
        init_args = list(init_signature.parameters.keys())[1:]  # Ignore 'self'

        ### Create a new instance, and give the constructor all the inputs it wants to see (based on values in this instance)
        new_op_point: __class__ = self.__class__(**{
            k: getattr(self, k)
            for k in init_args
        })

        ### Overwrite the state variables in the new instance with those from the input
        new_op_point._set_state(new_state=new_state)

        ### Return the new instance
        return new_op_point

    def _set_state(self,
                   new_state: Union[
                       Dict[str, Union[float, np.ndarray]],
                       List, Tuple, np.ndarray
                   ] = None
                   ):
        """
        Force-overwrites all state variables with a new set (either partial or complete) of state variables.

        Warning: this is *not* the intended public usage of OperatingPoint instances.
        If you want a new state yourself, you should instantiate a new one either:
            a) manually, or
            b) by using OperatingPoint.get_new_instance_with_state()

        Hence, this function is meant for PRIVATE use only - be careful how you use this!
        """
        ### Set the default parameters
        if new_state is None:
            new_state = {}

        try:  # Assume `value` is a dict-like, with keys
            for key in new_state.keys():  # Overwrite each of the specified state variables
                setattr(self, key, new_state[key])

        except AttributeError:  # Assume it's an iterable that has been sorted.
            self._set_state(
                self.pack_state(new_state))  # Pack the iterable into a dict-like, then do the same thing as above.

    def unpack_state(self,
                     dict_like_state: Dict[str, Union[float, np.ndarray]] = None
                     ) -> Tuple[Union[float, np.ndarray]]:
        """
        'Unpacks' a Dict-like state into an array-like that represents the state of the OperatingPoint.

        Args:
            dict_like_state: Takes in a dict-like representation of the state.

        Returns: The array representation of the state that you gave.

        """
        if dict_like_state is None:
            dict_like_state = self.state
        return tuple(dict_like_state.values())

    def pack_state(self,
                   array_like_state: Union[List, Tuple, np.ndarray] = None
                   ) -> Dict[str, Union[float, np.ndarray]]:
        """
        'Packs' an array into a Dict that represents the state of the OperatingPoint.

        Args:
            array_like_state: Takes in an iterable that must have the same number of entries as the state vector of the OperatingPoint.

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

    def __repr__(self) -> str:

        title = f"{self.__class__.__name__} instance:"

        def makeline(k, v):
            name = trim_string(str(k).strip(), length=10).rjust(10)
            item = trim_string(str(v).strip(), length=120).ljust(120)

            line = f"{name}: {item}"

            return line

        state_variables_title = "\tState variables:"

        state_variables = "\n".join([
            "\t\t" + makeline(k, v)
            for k, v in self.state.items()
        ])

        return "\n".join([
            title,
            state_variables_title,
            state_variables,
        ])

    def __getitem__(self, index: int) -> "OperatingPoint":
        """
        Indexes one item from each attribute of an OperatingPoint instance.
        Returns a new OperatingPoint instance.

        Args:
            index: The index that is being called; e.g.,:
                >>> first_op_point = op_point[0]

        Returns: A new OperatingPoint instance, where each attribute is subscripted at the given value, if possible.

        """
        l = len(self)
        if index >= l or index < -l:
            raise IndexError("Index is out of range!")

        def get_item_of_attribute(a):
            try:
                return a[index]
            except TypeError as e:  # object is not subscriptable
                return a
            except IndexError as e:  # index out of range
                raise IndexError("A state variable could not be indexed, since the index is out of range!")
            except NotImplementedError as e:
                raise TypeError(f"Indices must be integers or slices, not {index.__class__.__name__}")

        new_instance = self.get_new_instance_with_state()

        for k, v in new_instance.__dict__.items():
            setattr(new_instance, k, get_item_of_attribute(v))

        return new_instance

    def __len__(self):
        length = 0
        for v in self.state.values():
            if np.length(v) == 1:
                try:
                    v[0]
                    length = 1
                except (TypeError, IndexError, KeyError) as e:
                    pass
            elif length == 0 or length == 1:
                length = np.length(v)
            elif length == np.length(v):
                pass
            else:
                raise ValueError("State variables are appear vectorized, but of different lengths!")
        return length

    def __array__(self, dtype="O"):
        """
        Allows NumPy array creation without infinite recursion in __len__ and __getitem__.
        """
        return np.fromiter([self], dtype=dtype).reshape(())

    def dynamic_pressure(self):
        """
        Dynamic pressure of the working fluid
        Returns:
            float: Dynamic pressure of the working fluid. [Pa]
        """
        return 0.5 * self.atmosphere.density() * self.velocity ** 2

    def total_pressure(self):
        """
        Total (stagnation) pressure of the working fluid.

        Assumes a calorically perfect gas (i.e. specific heats do not change across the isentropic deceleration).

        Note that `total pressure != static pressure + dynamic pressure`, due to compressibility effects.

        Returns: Total pressure of the working fluid. [Pa]

        """
        gamma = self.atmosphere.ratio_of_specific_heats()
        return self.atmosphere.pressure() * (
                1 + (gamma - 1) / 2 * self.mach() ** 2
        ) ** (
                gamma / (gamma - 1)
        )

    def total_temperature(self):
        """
        Total (stagnation) temperature of the working fluid.

        Assumes a calorically perfect gas (i.e. specific heats do not change across the isentropic deceleration).

        Returns: Total temperature of the working fluid [K]

        """
        gamma = self.atmosphere.ratio_of_specific_heats()
        # return self.atmosphere.temperature() * (
        #         self.total_pressure() / self.atmosphere.pressure()
        # ) ** (
        #                (gamma - 1) / gamma
        #        )
        return self.atmosphere.temperature() * (
                1 + (gamma - 1) / 2 * self.mach() ** 2
        )

    def reynolds(self, reference_length):
        """
        Computes a Reynolds number with respect to a given reference length.
        :param reference_length: A reference length you choose [m]
        :return: Reynolds number [unitless]
        """
        density = self.atmosphere.density()
        viscosity = self.atmosphere.dynamic_viscosity()

        return density * self.velocity * reference_length / viscosity

    def mach(self):
        """
        Returns the Mach number associated with the current flight condition.
        """
        return self.velocity / self.atmosphere.speed_of_sound()

    def indicated_airspeed(self):
        """
        Returns the indicated airspeed associated with the current flight condition, in meters per second.
        """
        return np.sqrt(
            2 * (self.total_pressure() - self.atmosphere.pressure())
            / Atmosphere(altitude=0, method="isa").density()
        )

    def equivalent_airspeed(self):
        """
        Returns the equivalent airspeed associated with the current flight condition, in meters per second.
        """
        return self.velocity * np.sqrt(
            self.atmosphere.density() / Atmosphere(altitude=0, method="isa").density()
        )

    def energy_altitude(self):
        """
        Returns the energy altitude associated with the current flight condition, in meters.

        The energy altitude is the altitude at which a stationary aircraft would have the same total energy (kinetic
        + gravitational potential) as the aircraft at the current flight condition.

        """
        return self.atmosphere.altitude + 1 / (2 * 9.81) * self.velocity ** 2

    def convert_axes(self,
                     x_from: Union[float, np.ndarray],
                     y_from: Union[float, np.ndarray],
                     z_from: Union[float, np.ndarray],
                     from_axes: str,
                     to_axes: str,
                     ) -> Tuple[float, float, float]:
        """
        Converts a vector [x_from, y_from, z_from], as given in the `from_axes` frame, to an equivalent vector [x_to,
        y_to, z_to], as given in the `to_axes` frame.

        Both `from_axes` and `to_axes` should be a string, one of:
            * "geometry"
            * "body"
            * "wind"
            * "stability"

        This whole function is vectorized, both over the vector and the OperatingPoint (e.g., a vector of
        `OperatingPoint.alpha` values)

        Wind axes rotations are taken from Eq. 6.7 in Sect. 6.2.2 of Drela's Flight Vehicle Aerodynamics textbook,
        with axes corrections to go from [D, Y, L] to true wind axes (and same for geometry to body axes).

        Args:
            x_from: x-component of the vector, in `from_axes` frame.
            y_from: y-component of the vector, in `from_axes` frame.
            z_from: z-component of the vector, in `from_axes` frame.
            from_axes: The axes to convert from.
            to_axes: The axes to convert to.

        Returns: The x-, y-, and z-components of the vector, in `to_axes` frame. Given as a tuple.

        """
        if from_axes == to_axes:
            return x_from, y_from, z_from

        if from_axes == "geometry":
            x_b = -x_from
            y_b = y_from
            z_b = -z_from
        elif from_axes == "body":
            x_b = x_from
            y_b = y_from
            z_b = z_from
        elif from_axes == "wind":
            sa = np.sind(self.alpha)
            ca = np.cosd(self.alpha)
            sb = np.sind(self.beta)
            cb = np.cosd(self.beta)
            x_b = (cb * ca) * x_from + (-sb * ca) * y_from + (-sa) * z_from
            y_b = (sb) * x_from + (cb) * y_from  # Note: z term is 0; not forgotten.
            z_b = (cb * sa) * x_from + (-sb * sa) * y_from + (ca) * z_from
        elif from_axes == "stability":
            sa = np.sind(self.alpha)
            ca = np.cosd(self.alpha)
            x_b = ca * x_from - sa * z_from
            y_b = y_from
            z_b = sa * x_from + ca * z_from
        else:
            raise ValueError("Bad value of `from_axes`!")

        if to_axes == "geometry":
            x_to = -x_b
            y_to = y_b
            z_to = -z_b
        elif to_axes == "body":
            x_to = x_b
            y_to = y_b
            z_to = z_b
        elif to_axes == "wind":
            sa = np.sind(self.alpha)
            ca = np.cosd(self.alpha)
            sb = np.sind(self.beta)
            cb = np.cosd(self.beta)
            x_to = (cb * ca) * x_b + (sb) * y_b + (cb * sa) * z_b
            y_to = (-sb * ca) * x_b + (cb) * y_b + (-sb * sa) * z_b
            z_to = (-sa) * x_b + (ca) * z_b  # Note: y term is 0; not forgotten.
        elif to_axes == "stability":
            sa = np.sind(self.alpha)
            ca = np.cosd(self.alpha)
            x_to = ca * x_b + sa * z_b
            y_to = y_b
            z_to = -sa * x_b + ca * z_b
        else:
            raise ValueError("Bad value of `to_axes`!")

        return x_to, y_to, z_to

    def compute_rotation_matrix_wind_to_geometry(self) -> np.ndarray:
        """
        Computes the 3x3 rotation matrix that transforms from wind axes to geometry axes.

        Returns: a 3x3 rotation matrix.

        """

        alpha_rotation = np.rotation_matrix_3D(
            angle=np.radians(-self.alpha),
            axis="y",
        )
        beta_rotation = np.rotation_matrix_3D(
            angle=np.radians(self.beta),
            axis="z",
        )
        axes_flip = np.rotation_matrix_3D(
            angle=np.pi,
            axis="y",
        )
        # Since in geometry axes, X is downstream by convention, while in wind axes, X is upstream by convention.
        # Same with Z being up/down respectively.

        r = axes_flip @ alpha_rotation @ beta_rotation  # where "@" is the matrix multiplication operator

        return r

    def compute_freestream_direction_geometry_axes(self):
        # Computes the freestream direction (direction the wind is GOING TO) in the geometry axes
        return self.compute_rotation_matrix_wind_to_geometry() @ np.array([-1, 0, 0])

    def compute_freestream_velocity_geometry_axes(self):
        # Computes the freestream velocity vector (direction the wind is GOING TO) in geometry axes
        return self.compute_freestream_direction_geometry_axes() * self.velocity

    def compute_rotation_velocity_geometry_axes(self, points):
        # Computes the effective velocity-due-to-rotation at a set of points.
        # Input: a Nx3 array of points
        # Output: a Nx3 array of effective velocities
        angular_velocity_vector_geometry_axes = np.array([
            -self.p,
            self.q,
            -self.r
        ])  # signs convert from body axes to geometry axes

        a = angular_velocity_vector_geometry_axes
        b = points

        rotation_velocity_geometry_axes = np.stack([
            a[1] * b[:, 2] - a[2] * b[:, 1],
            a[2] * b[:, 0] - a[0] * b[:, 2],
            a[0] * b[:, 1] - a[1] * b[:, 0]
        ], axis=1)

        rotation_velocity_geometry_axes = -rotation_velocity_geometry_axes  # negative sign, since we care about the velocity the WING SEES, not the velocity of the wing.

        return rotation_velocity_geometry_axes


if __name__ == '__main__':
    op_point = OperatingPoint()
