import aerosandbox.numpy as np
from aerosandbox.common import AeroSandboxObject
from abc import ABC, abstractmethod, abstractproperty
from typing import Union, Dict, Tuple, List
from aerosandbox import MassProperties, Opti, OperatingPoint, Atmosphere, Airplane, _asb_root
from aerosandbox.tools.string_formatting import trim_string
import inspect
import copy


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
        """
        Returns the state variables of this Dynamics instance as a Dict.

        Keys are strings that give the name of the variables.
        Values are the variables themselves.

        This method should look something like:
            >>> {
            >>>     "x_e": self.x_e,
            >>>     "u_e": self.u_e,
            >>>     ...
            >>> }

        """
        pass

    def get_new_instance_with_state(self,
                                    new_state: Union[
                                        Dict[str, Union[float, np.ndarray]],
                                        List, Tuple, np.ndarray
                                    ] = None
                                    ):
        """
        Creates a new instance of this same Dynamics class from the given state.

        Note that any control variables (forces, moments) associated with the previous instance are zeroed.

        Args:
            new_state: The new state to be used for the new instance. Ideally, this is represented as a Dict in identical format to the `state` of a Dynamics instance.

        Returns: A new instance of this same Dynamics class.

        """

        ### Get a list of all the inputs that the class constructor wants to see
        init_signature = inspect.signature(self.__class__.__init__)
        init_args = list(init_signature.parameters.keys())[1:]  # Ignore 'self'

        ### Create a new instance, and give the constructor all the inputs it wants to see (based on values in this instance)
        new_dyn: __class__ = self.__class__(**{
            k: getattr(self, k)
            for k in init_args
        })

        ### Overwrite the state variables in the new instance with those from the input
        new_dyn._set_state(new_state=new_state)

        ### Return the new instance
        return new_dyn

    def _set_state(self,
                   new_state: Union[
                       Dict[str, Union[float, np.ndarray]],
                       List, Tuple, np.ndarray
                   ] = None
                   ):
        """
        Force-overwrites all state variables with a new set (either partial or complete) of state variables.

        Warning: this is *not* the intended public usage of Dynamics instances.
        If you want a new state yourself, you should instantiate a new one either:
            a) manually, or
            b) by using Dynamics.get_new_instance_with_state()

        Hence, this function is meant for PRIVATE use only - be careful how you use this! Especially note that
        control variables (e.g., forces, moments) do not reset to zero.
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
        'Unpacks' a Dict-like state into an array-like that represents the state of the dynamical system.

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
        """
        Indexes one item from each attribute of a Dynamics instance.
        Returns a new Dynamics instance.

        Args:
            index: The index that is being called; e.g.,:
                >>> first_dyn = dyn[0]

        Returns: A new Dynamics instance, where each attribute is subscripted at the given value, if possible.

        """

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
        length = 1
        for v in self.state.values():
            if np.length(v) == 1:
                pass
            elif length == 1:
                length = np.length(v)
            elif length == np.length(v):
                pass
            else:
                raise ValueError("State variables are appear vectorized, but of different lengths!")
        return length

    @abstractmethod
    def state_derivatives(self) -> Dict[str, Union[float, np.ndarray]]:
        """
        A function that returns the derivatives with respect to time of the state specified in the `state` property.

        Should return a Dict with the same keys as the `state` property.
        """
        pass

    def constrain_derivatives(self,
                              opti: Opti,
                              time: np.ndarray,
                              method: str = "midpoint",
                              which: Union[str, List[str], Tuple[str]] = "all"
                              ):
        """
        Applies the relevant state derivative constraints to a given Opti instance.

        Args:

            opti: the AeroSandbox `Opti` instance that constraints should be applied to.

            time: A vector that represents the time at each discrete point. Should be the same length as any
            vectorized state variables in the `state` property of this Dynamics instance.

            method: The discrete integration method to use. See Opti.constrain_derivative() for options.

            which: Which state variables should be we constrain? By default, constrains all of them.

                Options:

                    * "all", which constrains all state variables (default)

                    * A list of strings that are state variable names (i.e., a subset of `dyn.state.keys()`),
                    that gives the names of state variables to be constrained.

        Returns:

        """
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
                    method=method,
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
                  axes: str = "wind",
                  ) -> None:
        """
        Adds a force (in whichever axis system you choose) to this dynamics instance.

        Args:
            Fx: Force in the x-direction in the axis system chosen. [N]
            Fy: Force in the y-direction in the axis system chosen. [N]
            Fz: Force in the z-direction in the axis system chosen. [N]
            axes: The axis system that the specified force is in. One of:
                * "geometry"
                * "body"
                * "wind"
                * "stability"
                * "earth"

        Returns: None (in-place)

        """
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

    def draw(self,
             vehicle_model: Airplane = None,
             backend: str = "pyvista",
             draw_axes: bool = True,
             scale_vehicle_model: Union[float, None] = None,
             n_vehicles_to_draw: int = 10,
             cg_axes: str = "geometry",
             show: bool = True,
             ):
        if backend == "pyvista":
            import pyvista as pv
            import aerosandbox.tools.pretty_plots as p

            if vehicle_model is None:
                default_vehicle_stl = _asb_root / "dynamics/visualization/default_assets/yf23.stl"
                vehicle_model = pv.read(str(default_vehicle_stl))
            elif isinstance(vehicle_model, pv.PolyData):
                pass
            elif isinstance(vehicle_model, Airplane):
                vehicle_model = vehicle_model.draw(
                    backend="pyvista",
                    show=False
                )
                vehicle_model.rotate_y(180)  # Rotate from geometry axes to body axes.
            elif isinstance(vehicle_model, str):  # Interpret the string as a filepath to a .stl or similar
                try:
                    pv.read(filename=vehicle_model)
                except:
                    raise ValueError("Could not parse `vehicle_model`!")
            else:
                raise TypeError("`vehicle_model` should be an Airplane or PolyData object.")

            x_e = np.array(self.x_e)
            y_e = np.array(self.y_e)
            z_e = np.array(self.z_e)
            if np.length(x_e) == 1:
                x_e = x_e * np.ones(len(self))
            if np.length(y_e) == 1:
                y_e = y_e * np.ones(len(self))
            if np.length(z_e) == 1:
                z_e = z_e * np.ones(len(self))

            if scale_vehicle_model is None:
                trajectory_bounds = np.array([
                    [x_e.min(), x_e.max()],
                    [y_e.min(), y_e.max()],
                    [z_e.min(), z_e.max()],
                ])
                trajectory_size = np.max(np.diff(trajectory_bounds, axis=1))

                vehicle_bounds = np.array(vehicle_model.bounds).reshape((3, 2))
                vehicle_size = np.max(np.diff(vehicle_bounds, axis=1))

                scale_vehicle_model = 0.1 * trajectory_size / vehicle_size

            ### Initialize the plotter
            plotter = pv.Plotter()

            # Set the window title
            title = "ASB Dynamics"
            addenda = []
            if scale_vehicle_model != 1:
                addenda.append(f"Vehicle drawn at {scale_vehicle_model:.2g}x scale")
            addenda.append(f"{self.__class__.__name__} Engine")
            if len(addenda) != 0:
                title = title + f" ({'; '.join(addenda)})"
            plotter.title = title

            # Draw axes and grid
            plotter.add_axes()
            plotter.show_grid(color='gray')

            ### Draw the vehicle
            for i in np.unique(
                    np.round(
                        np.linspace(0, len(self) - 1, n_vehicles_to_draw)
                    )
            ).astype(int):
                dyn = self[i]
                try:
                    phi = dyn.phi
                except AttributeError:
                    phi = dyn.bank
                try:
                    theta = dyn.theta
                except AttributeError:
                    theta = dyn.gamma
                try:
                    psi = dyn.psi
                except AttributeError:
                    psi = dyn.track

                x_cg_b, y_cg_b, z_cg_b = dyn.convert_axes(
                    dyn.mass_props.x_cg,
                    dyn.mass_props.y_cg,
                    dyn.mass_props.z_cg,
                    from_axes=cg_axes,
                    to_axes="body"
                )

                this_vehicle = copy.deepcopy(vehicle_model)
                this_vehicle.translate([
                    -x_cg_b,
                    -y_cg_b,
                    -z_cg_b,
                ], inplace=True)
                this_vehicle.points *= scale_vehicle_model
                this_vehicle.rotate_x(np.degrees(phi), inplace=True)
                this_vehicle.rotate_y(np.degrees(theta), inplace=True)
                this_vehicle.rotate_z(np.degrees(psi), inplace=True)
                this_vehicle.translate([
                    dyn.x_e,
                    dyn.y_e,
                    dyn.z_e,
                ], inplace=True)
                plotter.add_mesh(
                    this_vehicle,
                )
                if draw_axes:
                    rot = np.rotation_matrix_from_euler_angles(phi, theta, psi)
                    axes_scale = 0.5 * np.max(
                        np.diff(
                            np.array(this_vehicle.bounds).reshape((3, -1)),
                            axis=1
                        )
                    )
                    origin = np.array([
                        dyn.x_e,
                        dyn.y_e,
                        dyn.z_e,
                    ])
                    for i, c in enumerate(["r", "g", "b"]):
                        plotter.add_mesh(
                            pv.Spline(np.array([
                                origin,
                                origin + rot[:, i] * axes_scale
                            ])),
                            color=c,
                            line_width=2.5,
                        )

            for i in range(len(self)):
                ### Draw the trajectory line

                polyline = pv.Spline(np.array([x_e, y_e, z_e]).T)
                plotter.add_mesh(
                    polyline,
                    color=p.adjust_lightness(p.palettes["categorical"][0], 1.2),
                    line_width=3,
                )

            ### Finalize the plotter
            plotter.camera.up = (0, 0, -1)
            plotter.camera.Azimuth(90)
            plotter.camera.Elevation(60)
            if show:
                plotter.show()
            return plotter

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
