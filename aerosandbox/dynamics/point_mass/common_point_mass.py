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

    @property
    @abstractmethod
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

    @property
    @abstractmethod
    def control_variables(self) -> Dict[str, Union[float, np.ndarray]]:
        pass

    def __repr__(self) -> str:

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

    def __getitem__(self, index: int):
        """
        Indexes one item from each attribute of a Dynamics instance.
        Returns a new Dynamics instance of the same type.

        Args:
            index: The index that is being called; e.g.,:
                >>> first_dyn = dyn[0]

        Returns: A new Dynamics instance, where each attribute is subscripted at the given value, if possible.

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
                              method: str = "trapezoidal",
                              which: Union[str, List[str], Tuple[str]] = "all",
                              _stacklevel=1,
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

            _stacklevel: Optional and advanced, purely used for debugging. Allows users to correctly track where
            constraints are declared in the event that they are subclassing `aerosandbox.Opti`. Modifies the
            stacklevel of the declaration tracked, which is then presented using
            `aerosandbox.Opti.variable_declaration()` and `aerosandbox.Opti.constraint_declaration()`.

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
                    _stacklevel=_stacklevel + 1
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

                from_axes: The axes to convert from. See above for options.

                to_axes: The axes to convert to. See above for options.


        Returns: The x-, y-, and z-components of the vector, in `to_axes` frame. Given as a tuple.

        """
        pass

    @abstractmethod
    def add_force(self,
                  Fx: Union[float, np.ndarray] = 0,
                  Fy: Union[float, np.ndarray] = 0,
                  Fz: Union[float, np.ndarray] = 0,
                  axes: str = "wind",
                  ) -> None:
        """
        Adds a force (in whichever axis system you choose) to this Dynamics instance.

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
        """
        In-place modifies the forces associated with this Dynamics instance: adds a force in the -z direction,
        equal to the weight of the aircraft.

        Args:
            g: The gravitational acceleration. [m/s^2]

        Returns: None (in-place)
        """
        self.add_force(
            Fz=self.mass_props.mass * g,
            axes="earth",
        )

    @property
    def op_point(self):
        """
        Returns an OperatingPoint object that represents the current state of the dynamics instance.

        This OperatingPoint object is effectively a subset of the state variables, and is used to compute aerodynamic
        forces and moments.
        """
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
             vehicle_model: Union[Airplane, "PolyData"] = None,
             backend: str = "pyvista",
             plotter=None,
             draw_axes: bool = True,
             draw_global_axes: bool = True,
             draw_global_grid: bool = True,
             scale_vehicle_model: Union[float, None] = None,
             n_vehicles_to_draw: int = 10,
             cg_axes: str = "geometry",
             draw_trajectory_line: bool = True,
             trajectory_line_color=None,
             draw_altitude_drape: bool = True,
             draw_ground_plane: bool = True,
             draw_wingtip_ribbon: bool = True,
             set_sky_background: bool = True,
             vehicle_color=None,
             show: bool = True,
             ):
        if backend == "pyvista":
            import pyvista as pv
            import aerosandbox.tools.pretty_plots as p

            if vehicle_model is None:
                default_vehicle_stl = _asb_root / "dynamics/visualization/default_assets/talon.stl"
                vehicle_model = pv.read(str(default_vehicle_stl))
            elif isinstance(vehicle_model, pv.PolyData):
                pass
            elif isinstance(vehicle_model, Airplane):
                vehicle_model: pv.PolyData = vehicle_model.draw(
                    backend="pyvista",
                    show=False
                )
                vehicle_model.rotate_y(180, inplace=True)  # Rotate from geometry axes to body axes.
            elif isinstance(vehicle_model, str):  # Interpret the string as a filepath to a .stl or similar
                try:
                    pv.read(filename=vehicle_model)
                except Exception:
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

            trajectory_bounds = np.array([
                [x_e.min(), x_e.max()],
                [y_e.min(), y_e.max()],
                [z_e.min(), z_e.max()],
            ])

            vehicle_bounds = np.array(vehicle_model.bounds).reshape((3, 2))

            # trajectory_size = np.max(np.diff(trajectory_bounds, axis=1))  # Max dimension
            # vehicle_size = np.max(np.diff(vehicle_bounds, axis=1))  # Max dimension

            if scale_vehicle_model is None:  # Compute an auto-scaling factor
                if len(self) == 1:
                    scale_vehicle_model = 1
                else:
                    path_length = np.sum(
                        (np.diff(x_e) ** 2 + np.diff(y_e) ** 2 + np.diff(z_e) ** 2) ** 0.5
                    )
                    vehicle_length = np.diff(vehicle_bounds[0, :])
                    scale_vehicle_model = float(0.5 * path_length / vehicle_length / n_vehicles_to_draw)

            ### Initialize the plotter
            if plotter is None:
                plotter = pv.Plotter()

            if set_sky_background:
                plotter.set_background(
                    color="#FFFFFF",
                    top="#A5B8D7",
                )

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
            if draw_global_axes:
                plotter.add_axes()
            if draw_global_grid:
                plotter.show_grid(color='gray')

            ### Set up interpolators for dynamics instances
            from scipy import interpolate
            state_interpolators = {
                k: interpolate.InterpolatedUnivariateSpline(
                    x=np.arange(len(self)),
                    y=v * np.ones(len(self)),
                    k=np.clip(len(self), 1, 3),
                    check_finite=True,
                )
                for k, v in self.state.items()
            }
            control_interpolators = {
                k: interpolate.InterpolatedUnivariateSpline(
                    x=np.arange(len(self)),
                    y=v * np.ones(len(self)),
                    k=np.clip(len(self), 1, 3),
                    check_finite=True,
                )
                for k, v in self.control_variables.items()
            }

            ### Draw the vehicle
            for i in np.linspace(0, len(self) - 1, n_vehicles_to_draw):
                dyn = self.get_new_instance_with_state({
                    k: float(v(i))
                    for k, v in state_interpolators.items()
                })
                for k, v in control_interpolators.items():
                    setattr(dyn, k, float(v(i)))

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
                    dyn.mass_props.x_cg,  # TODO fix this and make this per-point
                    dyn.mass_props.y_cg,
                    dyn.mass_props.z_cg,
                    from_axes=cg_axes,
                    to_axes="body"
                )

                this_vehicle = copy.deepcopy(vehicle_model)
                this_vehicle.translate([
                    -np.mean(x_cg_b),
                    -np.mean(y_cg_b),
                    -np.mean(z_cg_b),
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
                    color=(
                        p.adjust_lightness(p.palettes["categorical"][0], 1.3)
                        if vehicle_color is None
                        else vehicle_color
                    ),
                    opacity=0.95,
                    specular=0.5,
                    specular_power=15,
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
                            opacity=0.5,
                        )

            ### Draw the trajectory line
            path = np.stack([
                x_e,
                y_e,
                z_e,
            ], axis=1)

            if len(self) > 1:
                if draw_trajectory_line:
                    polyline = pv.Spline(path)
                    plotter.add_mesh(
                        polyline,
                        color=(
                            p.adjust_lightness(p.palettes["categorical"][0], 1.3)
                            if trajectory_line_color is None
                            else trajectory_line_color
                        ),
                        line_width=3,
                    )

                if draw_wingtip_ribbon:

                    left_wingtip_points = np.array(self.convert_axes(
                        0, scale_vehicle_model * vehicle_bounds[1, 0], 0,
                        from_axes="body",
                        to_axes="earth"
                    )).T + path
                    plotter.add_mesh(
                        pv.Spline(left_wingtip_points),
                        color="pink",
                    )
                    right_wingtip_points = np.array(self.convert_axes(
                        0, scale_vehicle_model * vehicle_bounds[1, 1], 0,
                        from_axes="body",
                        to_axes="earth"
                    )).T + path
                    plotter.add_mesh(
                        pv.Spline(right_wingtip_points),
                        color="pink",
                    )

                    grid = pv.StructuredGrid()
                    grid.points = np.concatenate([
                        left_wingtip_points,
                        right_wingtip_points,
                    ], axis=0)
                    grid.dimensions = len(left_wingtip_points), 2, 1

                    plotter.add_mesh(
                        grid,
                        color="pink",
                        opacity=0.5,
                    )

                if draw_altitude_drape:
                    ### Drape
                    grid = pv.StructuredGrid()
                    grid.points = np.concatenate([
                        path,
                        path * np.array([[1, 1, 0]])
                    ], axis=0)
                    grid.dimensions = len(path), 2, 1

                    plotter.add_mesh(
                        grid,
                        color="black",
                        opacity=0.5,
                    )

                if draw_ground_plane:
                    ### Plane
                    grid = pv.StructuredGrid()
                    xlim = (x_e.min(), x_e.max())
                    ylim = (y_e.min(), y_e.max())

                    grid.points = np.array([
                        [xlim[0], ylim[0], 0],
                        [xlim[1], ylim[0], 0],
                        [xlim[0], ylim[1], 0],
                        [xlim[1], ylim[1], 0]
                    ])
                    grid.dimensions = 2, 2, 1
                    plotter.add_mesh(
                        grid,
                        color="darkkhaki",
                        opacity=0.5
                    )

            ### Finalize the plotter
            plotter.camera.up = (0, 0, -1)
            plotter.camera.Azimuth(90)
            plotter.camera.Elevation(60)
            if show:
                plotter.show()
            return plotter

        else:
            raise NotImplementedError("Only the pyvista backend is implemented so far.")

    @property
    def altitude(self):
        return -self.z_e

    @property
    def translational_kinetic_energy(self) -> float:
        """
        Computes the kinetic energy [J] from translational motion.

        KE = 0.5 * m * v^2

        Returns:
            Kinetic energy [J]
        """
        return 0.5 * self.mass_props.mass * self.speed ** 2

    @property
    def rotational_kinetic_energy(self) -> float:
        """
        Computes the kinetic energy [J] from rotational motion.

        KE = 0.5 * I * w^2

        Returns:
            Kinetic energy [J]
        """
        return 0.5 * (
                self.mass_props.Ixx * self.p ** 2 +
                self.mass_props.Iyy * self.q ** 2 +
                self.mass_props.Izz * self.r ** 2
        )

    @property
    def kinetic_energy(self):
        """
        Computes the kinetic energy [J] from translational and rotational motion.

        KE = 0.5 * m * v^2 + 0.5 * I * w^2

        Returns:
            Kinetic energy [J]
        """
        return self.translational_kinetic_energy + self.rotational_kinetic_energy

    @property
    def potential_energy(self,
                         g: float = 9.81
                         ):
        """
        Computes the potential energy [J] from gravity.

        PE = mgh

        Args:
            g: Acceleration due to gravity [m/s^2]

        Returns:
            Potential energy [J]
        """
        return self.mass_props.mass * g * self.altitude
