import aerosandbox.numpy as np
from aerosandbox import ExplicitAnalysis
from aerosandbox.geometry import *
from aerosandbox.performance import OperatingPoint
from aerosandbox.aerodynamics.aero_3D.singularities.uniform_strength_horseshoe_singularities import \
    calculate_induced_velocity_horseshoe
from typing import Dict, Any, List, Callable
import copy


### Define some helper functions that take a vector and make it a Nx1 or 1xN, respectively.
# Useful for broadcasting with matrices later.
def tall(array):
    return np.reshape(array, (-1, 1))


def wide(array):
    return np.reshape(array, (1, -1))


class VortexLatticeMethod(ExplicitAnalysis):
    """
    An explicit (linear) vortex-lattice-method aerodynamics analysis.

    Usage example:
        >>> analysis = asb.VortexLatticeMethod(
        >>>     airplane=my_airplane,
        >>>     op_point=asb.OperatingPoint(
        >>>         velocity=100, # m/s
        >>>         alpha=5, # deg
        >>>         beta=4, # deg
        >>>         p=0.01, # rad/sec
        >>>         q=0.02, # rad/sec
        >>>         r=0.03, # rad/sec
        >>>     )
        >>> )
        >>> aero_data = analysis.run()
        >>> analysis.draw()
    """

    def __init__(self,
                 airplane: Airplane,
                 op_point: OperatingPoint,
                 xyz_ref: List[float] = None,
                 run_symmetric_if_possible: bool = False,
                 verbose: bool = False,
                 spanwise_resolution: int = 10,
                 spanwise_spacing_function: Callable[[float, float, float], np.ndarray] = np.cosspace,
                 chordwise_resolution: int = 10,
                 chordwise_spacing_function: Callable[[float, float, float], np.ndarray] = np.cosspace,
                 vortex_core_radius: float = 1e-8,
                 align_trailing_vortices_with_wind: bool = False,
                 ):
        super().__init__()

        ### Set defaults
        if xyz_ref is None:
            xyz_ref = airplane.xyz_ref

        ### Initialize
        self.airplane = airplane
        self.op_point = op_point
        self.xyz_ref = xyz_ref
        self.verbose = verbose
        self.spanwise_resolution = spanwise_resolution
        self.spanwise_spacing_function = spanwise_spacing_function
        self.chordwise_resolution = chordwise_resolution
        self.chordwise_spacing_function = chordwise_spacing_function
        self.vortex_core_radius = vortex_core_radius
        self.align_trailing_vortices_with_wind = align_trailing_vortices_with_wind

        ### Determine whether you should run the problem as symmetric
        self.run_symmetric = False
        if run_symmetric_if_possible:
            raise NotImplementedError("VLM with symmetry detection not yet implemented!")
            # try:
            #     self.run_symmetric = (  # Satisfies assumptions
            #             self.op_point.beta == 0 and
            #             self.op_point.p == 0 and
            #             self.op_point.r == 0 and
            #             self.airplane.is_entirely_symmetric()
            #     )
            # except RuntimeError:  # Required because beta, p, r, etc. may be non-numeric (e.g. opti variables)
            #     pass

    def __repr__(self):
        return self.__class__.__name__ + "(\n\t" + "\n\t".join([
            f"airplane={self.airplane}",
            f"op_point={self.op_point}",
            f"xyz_ref={self.xyz_ref}",
        ]) + "\n)"

    def run(self) -> Dict[str, Any]:
        """
        Computes the aerodynamic forces.

        Returns a dictionary with keys:

            - 'F_g' : an [x, y, z] list of forces in geometry axes [N]
            - 'F_b' : an [x, y, z] list of forces in body axes [N]
            - 'F_w' : an [x, y, z] list of forces in wind axes [N]
            - 'M_g' : an [x, y, z] list of moments about geometry axes [Nm]
            - 'M_b' : an [x, y, z] list of moments about body axes [Nm]
            - 'M_w' : an [x, y, z] list of moments about wind axes [Nm]
            - 'L' : the lift force [N]. Definitionally, this is in wind axes.
            - 'Y' : the side force [N]. This is in wind axes.
            - 'D' : the drag force [N]. Definitionally, this is in wind axes.
            - 'l_b', the rolling moment, in body axes [Nm]. Positive is roll-right.
            - 'm_b', the pitching moment, in body axes [Nm]. Positive is pitch-up.
            - 'n_b', the yawing moment, in body axes [Nm]. Positive is nose-right.
            - 'CL', the lift coefficient [-]. Definitionally, this is in wind axes.
            - 'CY', the sideforce coefficient [-]. This is in wind axes.
            - 'CD', the drag coefficient [-]. Definitionally, this is in wind axes.
            - 'Cl', the rolling coefficient [-], in body axes
            - 'Cm', the pitching coefficient [-], in body axes
            - 'Cn', the yawing coefficient [-], in body axes

        Nondimensional values are nondimensionalized using reference values in the VortexLatticeMethod.airplane object.
        """

        if self.verbose:
            print("Meshing...")

        ##### Make Panels
        front_left_vertices = []
        back_left_vertices = []
        back_right_vertices = []
        front_right_vertices = []
        is_trailing_edge = []

        for wing in self.airplane.wings:
            if self.spanwise_resolution > 1:
                wing = wing.subdivide_sections(
                    ratio=self.spanwise_resolution,
                    spacing_function=self.spanwise_spacing_function
                )

            points, faces = wing.mesh_thin_surface(
                method="quad",
                chordwise_resolution=self.chordwise_resolution,
                chordwise_spacing_function=self.chordwise_spacing_function,
                add_camber=True
            )
            front_left_vertices.append(points[faces[:, 0], :])
            back_left_vertices.append(points[faces[:, 1], :])
            back_right_vertices.append(points[faces[:, 2], :])
            front_right_vertices.append(points[faces[:, 3], :])
            is_trailing_edge.append(
                (np.arange(len(faces)) + 1) % self.chordwise_resolution == 0
            )

        front_left_vertices = np.concatenate(front_left_vertices)
        back_left_vertices = np.concatenate(back_left_vertices)
        back_right_vertices = np.concatenate(back_right_vertices)
        front_right_vertices = np.concatenate(front_right_vertices)
        is_trailing_edge = np.concatenate(is_trailing_edge)

        ### Compute panel statistics
        diag1 = front_right_vertices - back_left_vertices
        diag2 = front_left_vertices - back_right_vertices
        cross = np.cross(diag1, diag2)
        cross_norm = np.linalg.norm(cross, axis=1)
        normal_directions = cross / tall(cross_norm)
        areas = cross_norm / 2

        # Compute the location of points of interest on each panel
        left_vortex_vertices = 0.75 * front_left_vertices + 0.25 * back_left_vertices
        right_vortex_vertices = 0.75 * front_right_vertices + 0.25 * back_right_vertices
        vortex_centers = (left_vortex_vertices + right_vortex_vertices) / 2
        vortex_bound_leg = right_vortex_vertices - left_vortex_vertices
        collocation_points = (
                0.5 * (0.25 * front_left_vertices + 0.75 * back_left_vertices) +
                0.5 * (0.25 * front_right_vertices + 0.75 * back_right_vertices)
        )

        ### Save things to the instance for later access
        self.front_left_vertices = front_left_vertices
        self.back_left_vertices = back_left_vertices
        self.back_right_vertices = back_right_vertices
        self.front_right_vertices = front_right_vertices
        self.is_trailing_edge = is_trailing_edge
        self.normal_directions = normal_directions
        self.areas = areas
        self.left_vortex_vertices = left_vortex_vertices
        self.right_vortex_vertices = right_vortex_vertices
        self.vortex_centers = vortex_centers
        self.vortex_bound_leg = vortex_bound_leg
        self.collocation_points = collocation_points

        ##### Setup Operating Point
        if self.verbose:
            print("Calculating the freestream influence...")
        steady_freestream_velocity = self.op_point.compute_freestream_velocity_geometry_axes()  # Direction the wind is GOING TO, in geometry axes coordinates
        steady_freestream_direction = steady_freestream_velocity / np.linalg.norm(steady_freestream_velocity)
        rotation_freestream_velocities = self.op_point.compute_rotation_velocity_geometry_axes(
            collocation_points)

        freestream_velocities = np.add(wide(steady_freestream_velocity), rotation_freestream_velocities)
        # Nx3, represents the freestream velocity at each panel collocation point (c)

        freestream_influences = np.sum(freestream_velocities * normal_directions, axis=1)

        ### Save things to the instance for later access
        self.steady_freestream_velocity = steady_freestream_velocity
        self.steady_freestream_direction = steady_freestream_direction
        self.freestream_velocities = freestream_velocities

        ##### Setup Geometry
        ### Calculate AIC matrix
        if self.verbose:
            print("Calculating the collocation influence matrix...")

        u_collocations_unit, v_collocations_unit, w_collocations_unit = calculate_induced_velocity_horseshoe(
            x_field=tall(collocation_points[:, 0]),
            y_field=tall(collocation_points[:, 1]),
            z_field=tall(collocation_points[:, 2]),
            x_left=wide(left_vortex_vertices[:, 0]),
            y_left=wide(left_vortex_vertices[:, 1]),
            z_left=wide(left_vortex_vertices[:, 2]),
            x_right=wide(right_vortex_vertices[:, 0]),
            y_right=wide(right_vortex_vertices[:, 1]),
            z_right=wide(right_vortex_vertices[:, 2]),
            trailing_vortex_direction=(
                steady_freestream_direction
                if self.align_trailing_vortices_with_wind else
                np.array([1, 0, 0])
            ),
            gamma=1.,
            vortex_core_radius=self.vortex_core_radius
        )

        AIC = (
                u_collocations_unit * tall(normal_directions[:, 0]) +
                v_collocations_unit * tall(normal_directions[:, 1]) +
                w_collocations_unit * tall(normal_directions[:, 2])
        )

        ##### Calculate Vortex Strengths
        if self.verbose:
            print("Calculating vortex strengths...")

        self.vortex_strengths = np.linalg.solve(AIC, -freestream_influences)

        ##### Calculate forces
        ### Calculate Near-Field Forces and Moments
        # Governing Equation: The force on a straight, small vortex filament is F = rho * cross(V, l) * gamma,
        # where rho is density, V is the velocity vector, cross() is the cross product operator,
        # l is the vector of the filament itself, and gamma is the circulation.

        if self.verbose:
            print("Calculating forces on each panel...")
        # Calculate the induced velocity at the center of each bound leg
        V_centers = self.get_velocity_at_points(vortex_centers)

        # Calculate forces_inviscid_geometry, the force on the ith panel. Note that this is in GEOMETRY AXES,
        # not WIND AXES or BODY AXES.
        Vi_cross_li = np.cross(V_centers, vortex_bound_leg, axis=1)

        forces_geometry = self.op_point.atmosphere.density() * Vi_cross_li * tall(self.vortex_strengths)
        moments_geometry = np.cross(
            np.add(vortex_centers, -wide(np.array(self.xyz_ref))),
            forces_geometry
        )

        # Calculate total forces and moments
        force_geometry = np.sum(forces_geometry, axis=0)
        moment_geometry = np.sum(moments_geometry, axis=0)

        force_body = self.op_point.convert_axes(
            force_geometry[0], force_geometry[1], force_geometry[2],
            from_axes="geometry",
            to_axes="body"
        )
        force_wind = self.op_point.convert_axes(
            force_body[0], force_body[1], force_body[2],
            from_axes="body",
            to_axes="wind"
        )
        moment_body = self.op_point.convert_axes(
            moment_geometry[0], moment_geometry[1], moment_geometry[2],
            from_axes="geometry",
            to_axes="body"
        )
        moment_wind = self.op_point.convert_axes(
            moment_body[0], moment_body[1], moment_body[2],
            from_axes="body",
            to_axes="wind"
        )

        ### Save things to the instance for later access
        self.forces_geometry = forces_geometry
        self.moments_geometry = moments_geometry
        self.force_geometry = force_geometry
        self.force_body = force_body
        self.force_wind = force_wind
        self.moment_geometry = moment_geometry
        self.moment_body = moment_body
        self.moment_wind = moment_wind

        # Calculate dimensional forces
        L = -force_wind[2]
        D = -force_wind[0]
        Y = force_wind[1]
        l_b = moment_body[0]
        m_b = moment_body[1]
        n_b = moment_body[2]

        # Calculate nondimensional forces
        q = self.op_point.dynamic_pressure()
        s_ref = self.airplane.s_ref
        b_ref = self.airplane.b_ref
        c_ref = self.airplane.c_ref
        CL = L / q / s_ref
        CD = D / q / s_ref
        CY = Y / q / s_ref
        Cl = l_b / q / s_ref / b_ref
        Cm = m_b / q / s_ref / c_ref
        Cn = n_b / q / s_ref / b_ref

        return {
            "F_g": force_geometry,
            "F_b": force_body,
            "F_w": force_wind,
            "M_g": moment_geometry,
            "M_b": moment_body,
            "M_w": moment_wind,
            "L"  : L,
            "D"  : D,
            "Y"  : Y,
            "l_b": l_b,
            "m_b": m_b,
            "n_b": n_b,
            "CL" : CL,
            "CD" : CD,
            "CY" : CY,
            "Cl" : Cl,
            "Cm" : Cm,
            "Cn" : Cn,
        }

    def run_with_stability_derivatives(self,
                                       alpha=True,
                                       beta=True,
                                       p=True,
                                       q=True,
                                       r=True,
                                       ):
        """
                Computes the aerodynamic forces and moments on the airplane, and the stability derivatives.

                Arguments essentially determine which stability derivatives are computed. If a stability derivative is not
                needed, leaving it False will speed up the computation.

                Args:

                    - alpha (bool): If True, compute the stability derivatives with respect to the angle of attack (alpha).
                    - beta (bool): If True, compute the stability derivatives with respect to the sideslip angle (beta).
                    - p (bool): If True, compute the stability derivatives with respect to the body-axis roll rate (p).
                    - q (bool): If True, compute the stability derivatives with respect to the body-axis pitch rate (q).
                    - r (bool): If True, compute the stability derivatives with respect to the body-axis yaw rate (r).

                Returns: a dictionary with keys:

                    - 'F_g' : an [x, y, z] list of forces in geometry axes [N]
                    - 'F_b' : an [x, y, z] list of forces in body axes [N]
                    - 'F_w' : an [x, y, z] list of forces in wind axes [N]
                    - 'M_g' : an [x, y, z] list of moments about geometry axes [Nm]
                    - 'M_b' : an [x, y, z] list of moments about body axes [Nm]
                    - 'M_w' : an [x, y, z] list of moments about wind axes [Nm]
                    - 'L' : the lift force [N]. Definitionally, this is in wind axes.
                    - 'Y' : the side force [N]. This is in wind axes.
                    - 'D' : the drag force [N]. Definitionally, this is in wind axes.
                    - 'l_b', the rolling moment, in body axes [Nm]. Positive is roll-right.
                    - 'm_b', the pitching moment, in body axes [Nm]. Positive is pitch-up.
                    - 'n_b', the yawing moment, in body axes [Nm]. Positive is nose-right.
                    - 'CL', the lift coefficient [-]. Definitionally, this is in wind axes.
                    - 'CY', the sideforce coefficient [-]. This is in wind axes.
                    - 'CD', the drag coefficient [-]. Definitionally, this is in wind axes.
                    - 'Cl', the rolling coefficient [-], in body axes
                    - 'Cm', the pitching coefficient [-], in body axes
                    - 'Cn', the yawing coefficient [-], in body axes

                    Along with additional keys, depending on the value of the `alpha`, `beta`, `p`, `q`, and `r` arguments. For
                    example, if `alpha=True`, then the following additional keys will be present:

                        - 'CLa', the lift coefficient derivative with respect to alpha [1/rad]
                        - 'CDa', the drag coefficient derivative with respect to alpha [1/rad]
                        - 'CYa', the sideforce coefficient derivative with respect to alpha [1/rad]
                        - 'Cla', the rolling moment coefficient derivative with respect to alpha [1/rad]
                        - 'Cma', the pitching moment coefficient derivative with respect to alpha [1/rad]
                        - 'Cna', the yawing moment coefficient derivative with respect to alpha [1/rad]
                        - 'x_np', the neutral point location in the x direction [m]

                    Nondimensional values are nondimensionalized using reference values in the
                    VortexLatticeMethod.airplane object.

                    Data types:
                        - The "L", "Y", "D", "l_b", "m_b", "n_b", "CL", "CY", "CD", "Cl", "Cm", and "Cn" keys are:

                            - floats if the OperatingPoint object is not vectorized (i.e., if all attributes of OperatingPoint
                            are floats, not arrays).

                            - arrays if the OperatingPoint object is vectorized (i.e., if any attribute of OperatingPoint is an
                            array).

                        - The "F_g", "F_b", "F_w", "M_g", "M_b", and "M_w" keys are always lists, which will contain either
                        floats or arrays, again depending on whether the OperatingPoint object is vectorized or not.

                """
        abbreviations = {
            "alpha": "a",
            "beta" : "b",
            "p"    : "p",
            "q"    : "q",
            "r"    : "r",
        }
        finite_difference_amounts = {
            "alpha": 0.001,
            "beta" : 0.001,
            "p"    : 0.001 * (2 * self.op_point.velocity) / self.airplane.b_ref,
            "q"    : 0.001 * (2 * self.op_point.velocity) / self.airplane.c_ref,
            "r"    : 0.001 * (2 * self.op_point.velocity) / self.airplane.b_ref,
        }
        scaling_factors = {
            "alpha": np.degrees(1),
            "beta" : np.degrees(1),
            "p"    : (2 * self.op_point.velocity) / self.airplane.b_ref,
            "q"    : (2 * self.op_point.velocity) / self.airplane.c_ref,
            "r"    : (2 * self.op_point.velocity) / self.airplane.b_ref,
        }

        original_op_point = self.op_point

        # Compute the point analysis, which returns a dictionary that we will later add key:value pairs to.
        run_base = self.run()

        # Note for the loops below: here, "derivative numerator" and "... denominator" refer to the quantity being
        # differentiated and the variable of differentiation, respectively. In other words, in the expression df/dx,
        # the "numerator" is f, and the "denominator" is x. I realize that this would make a mathematician cry (as a
        # partial derivative is not a fraction), but the reality is that there seems to be no commonly-accepted name
        # for these terms. (Curiously, this contrasts with integration, where there is an "integrand" and a "variable
        # of integration".)

        for derivative_denominator in abbreviations.keys():
            if not locals()[derivative_denominator]:  # Basically, if the parameter from the function input is not True,
                continue  # Skip this run.
                # This way, you can (optionally) speed up this routine if you only need static derivatives,
                # or longitudinal derivatives, etc.

            # These lines make a copy of the original operating point, incremented by the finite difference amount
            # along the variable defined by derivative_denominator.
            incremented_op_point = copy.copy(original_op_point)
            incremented_op_point.__setattr__(
                derivative_denominator,
                original_op_point.__getattribute__(derivative_denominator) + finite_difference_amounts[
                    derivative_denominator]
            )

            vlm_incremented = copy.copy(self)
            vlm_incremented.op_point = incremented_op_point
            run_incremented = vlm_incremented.run()

            for derivative_numerator in [
                "CL",
                "CD",
                "CY",
                "Cl",
                "Cm",
                "Cn",
            ]:
                derivative_name = derivative_numerator + abbreviations[derivative_denominator]  # Gives "CLa"
                run_base[derivative_name] = (
                        (  # Finite-difference out the derivatives
                                run_incremented[derivative_numerator] - run_base[
                            derivative_numerator]
                        ) / finite_difference_amounts[derivative_denominator]
                        * scaling_factors[derivative_denominator]
                )

            ### Try to compute and append neutral point, if possible
            if derivative_denominator == "alpha":
                run_base["x_np"] = self.xyz_ref[0] - (
                        run_base["Cma"] * (self.airplane.c_ref / run_base["CLa"])
                )
            if derivative_denominator == "beta":
                run_base["x_np_lateral"] = self.xyz_ref[0] - (
                        run_base["Cnb"] * (self.airplane.b_ref / run_base["CYb"])
                )

        return run_base

    def get_induced_velocity_at_points(self,
                                       points: np.ndarray,
                                       ) -> np.ndarray:
        """
        Computes the induced velocity at a set of points in the flowfield.

        Args:
            points: A Nx3 array of points that you would like to know the induced velocities at. Given in geometry axes.

        Returns: A Nx3 of the induced velocity at those points. Given in geometry axes.

        """
        u_induced, v_induced, w_induced = calculate_induced_velocity_horseshoe(
            x_field=tall(points[:, 0]),
            y_field=tall(points[:, 1]),
            z_field=tall(points[:, 2]),
            x_left=wide(self.left_vortex_vertices[:, 0]),
            y_left=wide(self.left_vortex_vertices[:, 1]),
            z_left=wide(self.left_vortex_vertices[:, 2]),
            x_right=wide(self.right_vortex_vertices[:, 0]),
            y_right=wide(self.right_vortex_vertices[:, 1]),
            z_right=wide(self.right_vortex_vertices[:, 2]),
            trailing_vortex_direction=self.steady_freestream_direction if self.align_trailing_vortices_with_wind else np.array(
                [1, 0, 0]),
            gamma=wide(self.vortex_strengths),
            vortex_core_radius=self.vortex_core_radius
        )
        u_induced = np.sum(u_induced, axis=1)
        v_induced = np.sum(v_induced, axis=1)
        w_induced = np.sum(w_induced, axis=1)

        V_induced = np.stack([
            u_induced, v_induced, w_induced
        ], axis=1)

        return V_induced

    def get_velocity_at_points(self,
                               points: np.ndarray
                               ) -> np.ndarray:
        """
        Computes the velocity at a set of points in the flowfield.

        Args:
            points: A Nx3 array of points that you would like to know the velocities at. Given in geometry axes.

        Returns: A Nx3 of the velocity at those points. Given in geometry axes.

        """
        V_induced = self.get_induced_velocity_at_points(points)

        rotation_freestream_velocities = self.op_point.compute_rotation_velocity_geometry_axes(
            points
        )

        freestream_velocities = np.add(wide(self.steady_freestream_velocity), rotation_freestream_velocities)

        V = V_induced + freestream_velocities
        return V

    def calculate_streamlines(self,
                              seed_points: np.ndarray = None,
                              n_steps: int = 300,
                              length: float = None,
                              ) -> np.ndarray:
        """
        Computes streamlines, starting at specific seed points.

        After running this function, a new instance variable `VortexLatticeFilaments.streamlines` is computed

        Uses simple forward-Euler integration with a fixed spatial stepsize (i.e., velocity vectors are normalized
        before ODE integration). After investigation, it's not worth doing fancier ODE integration methods (adaptive
        schemes, RK substepping, etc.), due to the near-singular conditions near vortex filaments.

        Args:

            seed_points: A Nx3 ndarray that contains a list of points where streamlines are started. Will be
            auto-calculated if not specified.

            n_steps: The number of individual streamline steps to trace. Minimum of 2.

            length: The approximate total length of the streamlines desired, in meters. Will be auto-calculated if
            not specified.

        Returns:
            streamlines: a 3D array with dimensions: (n_seed_points) x (3) x (n_steps).
            Consists of streamlines data.

            Result is also saved as an instance variable, VortexLatticeMethod.streamlines.

        """
        if self.verbose:
            print("Calculating streamlines...")
        if length is None:
            length = self.airplane.c_ref * 5
        if seed_points is None:
            left_TE_vertices = self.back_left_vertices[self.is_trailing_edge.astype(bool)]
            right_TE_vertices = self.back_right_vertices[self.is_trailing_edge.astype(bool)]
            N_streamlines_target = 200
            seed_points_per_panel = np.maximum(1, N_streamlines_target // len(left_TE_vertices))

            nondim_node_locations = np.linspace(0, 1, seed_points_per_panel + 1)
            nondim_seed_locations = (nondim_node_locations[1:] + nondim_node_locations[:-1]) / 2

            seed_points = np.concatenate([
                x * left_TE_vertices + (1 - x) * right_TE_vertices
                for x in nondim_seed_locations
            ])

        streamlines = np.empty((len(seed_points), 3, n_steps))
        streamlines[:, :, 0] = seed_points
        for i in range(1, n_steps):
            V = self.get_velocity_at_points(streamlines[:, :, i - 1])
            streamlines[:, :, i] = (
                    streamlines[:, :, i - 1] +
                    length / n_steps * V / tall(np.linalg.norm(V, axis=1))
            )

        self.streamlines = streamlines

        if self.verbose:
            print("Streamlines calculated.")

        return streamlines

    def draw(self,
             c: np.ndarray = None,
             cmap: str = None,
             colorbar_label: str = None,
             show: bool = True,
             show_kwargs: Dict = None,
             draw_streamlines=True,
             recalculate_streamlines=False,
             backend: str = "pyvista"
             ):
        """
        Draws the solution. Note: Must be called on a SOLVED AeroProblem object.
        To solve an AeroProblem, use opti.solve(). To substitute a solved solution, use ap = sol(ap).
        :return:
        """
        if show_kwargs is None:
            show_kwargs = {}

        if c is None:
            c = self.vortex_strengths
            colorbar_label = "Vortex Strengths"

        if draw_streamlines:
            if (not hasattr(self, 'streamlines')) or recalculate_streamlines:
                self.calculate_streamlines()

        if backend == "plotly":
            from aerosandbox.visualization.plotly_Figure3D import Figure3D
            fig = Figure3D()

            for i in range(len(self.front_left_vertices)):
                fig.add_quad(
                    points=[
                        self.front_left_vertices[i, :],
                        self.back_left_vertices[i, :],
                        self.back_right_vertices[i, :],
                        self.front_right_vertices[i, :],
                    ],
                    intensity=c[i],
                    outline=True,
                )

            if draw_streamlines:
                for i in range(self.streamlines.shape[0]):
                    fig.add_streamline(self.streamlines[i, :, :].T)

            return fig.draw(
                show=show,
                colorbar_title=colorbar_label,
                **show_kwargs,
            )

        elif backend == "pyvista":
            import pyvista as pv
            plotter = pv.Plotter()
            plotter.title = "ASB VortexLatticeMethod"
            plotter.add_axes()
            plotter.show_grid(color='gray')

            ### Draw the airplane mesh
            points = np.concatenate([
                self.front_left_vertices,
                self.back_left_vertices,
                self.back_right_vertices,
                self.front_right_vertices
            ])
            N = len(self.front_left_vertices)
            range_N = np.arange(N)
            faces = tall(range_N) + wide(np.array([0, 1, 2, 3]) * N)

            mesh = pv.PolyData(
                *mesh_utils.convert_mesh_to_polydata_format(points, faces)
            )
            scalar_bar_args = {}
            if colorbar_label is not None:
                scalar_bar_args["title"] = colorbar_label
            plotter.add_mesh(
                mesh=mesh,
                scalars=c,
                show_edges=True,
                show_scalar_bar=c is not None,
                scalar_bar_args=scalar_bar_args,
                cmap=cmap,
            )

            ### Draw the streamlines
            if draw_streamlines:
                import aerosandbox.tools.pretty_plots as p
                for i in range(self.streamlines.shape[0]):
                    plotter.add_mesh(
                        pv.Spline(self.streamlines[i, :, :].T),
                        color=p.adjust_lightness("#7700FF", 1.5),
                        opacity=0.7,
                        line_width=1
                    )

            if show:
                plotter.show(**show_kwargs)
            return plotter

        else:
            raise ValueError("Bad value of `backend`!")


if __name__ == '__main__':
    ### Import Vanilla Airplane
    import aerosandbox as asb

    from pathlib import Path

    geometry_folder = Path(__file__).parent / "test_aero_3D" / "geometries"

    import sys

    sys.path.insert(0, str(geometry_folder))

    from vanilla import airplane as vanilla

    ### Do the AVL run
    vlm = VortexLatticeMethod(
        airplane=vanilla,
        op_point=asb.OperatingPoint(
            atmosphere=asb.Atmosphere(altitude=0),
            velocity=10,
            alpha=0,
            beta=0,
            p=0,
            q=0,
            r=0,
        ),
        spanwise_resolution=12,
        chordwise_resolution=12,
    )

    res = vlm.run()

    for k, v in res.items():
        print(f"{str(k).rjust(10)} : {v}")
