from aerosandbox import ExplicitAnalysis
from aerosandbox.geometry import *
from aerosandbox.performance import OperatingPoint
from aerosandbox.aerodynamics.aero_3D.singularities.uniform_strength_horseshoe_singularities import \
    calculate_induced_velocity_horseshoe
from aerosandbox.aerodynamics.aero_3D.singularities.point_source import calculate_induced_velocity_point_source
from typing import Dict, Any, List, Callable, Union
from aerosandbox.aerodynamics.aero_3D.aero_buildup import AeroBuildup
from dataclasses import dataclass


### Define some helper functions that take a vector and make it a Nx1 or 1xN, respectively.
# Useful for broadcasting with matrices later.
def tall(array):
    return np.reshape(array, (-1, 1))


def wide(array):
    return np.reshape(array, (1, -1))


class LiftingLine(ExplicitAnalysis):
    """
    An implicit aerodynamics analysis based on lifting line theory, with modifications for nonzero sweep
    and dihedral + multiple wings.

    Nonlinear, and includes viscous effects based on 2D data.

    Usage example:
        >>> analysis = asb.LiftingLine(
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
        >>> outputs = analysis.run()
    """

    def __init__(self,
                 airplane: Airplane,
                 op_point: OperatingPoint,
                 xyz_ref: List[float] = None,
                 model_size: str = "medium",
                 run_symmetric_if_possible: bool = False,
                 verbose: bool = False,
                 spanwise_resolution: int = 4,
                 spanwise_spacing_function: Callable[[float, float, float], np.ndarray] = np.cosspace,
                 vortex_core_radius: float = 1e-8,
                 align_trailing_vortices_with_wind: bool = False,
                 ):
        """
        Initializes and conducts a LiftingLine analysis.

        Args:

            airplane: An Airplane object that you want to analyze.

            op_point: The OperatingPoint that you want to analyze the Airplane at.

            run_symmetric_if_possible: If this flag is True and the problem fomulation is XZ-symmetric, the solver will
            attempt to exploit the symmetry. This results in roughly half the number of governing equations.

            opti: An asb.Opti environment.

                If provided, adds the governing equations to that instance. Does not solve the equations (you need to
                call `sol = opti.solve()` to do that).

                If not provided, creates and solves the governing equations in a new instance.

        """
        super().__init__()

        ### Set defaults
        if xyz_ref is None:
            xyz_ref = airplane.xyz_ref

        ### Initialize
        self.airplane = airplane
        self.op_point = op_point
        self.xyz_ref = xyz_ref
        self.model_size = model_size
        self.verbose = verbose
        self.spanwise_resolution = spanwise_resolution
        self.spanwise_spacing_function = spanwise_spacing_function
        self.vortex_core_radius = vortex_core_radius
        self.align_trailing_vortices_with_wind = align_trailing_vortices_with_wind

        ### Determine whether you should run the problem as symmetric
        self.run_symmetric = False
        if run_symmetric_if_possible:
            raise NotImplementedError("LiftingLine with symmetry detection not yet implemented!")
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

    @dataclass
    class AeroComponentResults:
        s_ref: float  # Reference area [m^2]
        c_ref: float  # Reference chord [m]
        b_ref: float  # Reference span [m]
        op_point: OperatingPoint
        F_g: List[Union[float, np.ndarray]]  # An [x, y, z] list of forces in geometry axes [N]
        M_g: List[Union[float, np.ndarray]]  # An [x, y, z] list of moments about geometry axes [Nm]

        def __repr__(self):
            F_w = self.F_w
            M_b = self.M_b
            return self.__class__.__name__ + "(\n\t" + "\n\t".join([
                f"L={-F_w[2]},",
                f"Y={F_w[1]},",
                f"D={-F_w[0]},",
                f"l_b={M_b[0]},",
                f"m_b={M_b[1]},",
                f"n_b={M_b[2]},",
            ]) + "\n)"

        @property
        def F_b(self) -> List[Union[float, np.ndarray]]:
            """
            An [x, y, z] list of forces in body axes [N]
            """
            return self.op_point.convert_axes(*self.F_g, from_axes="geometry", to_axes="body")

        @property
        def F_w(self) -> List[Union[float, np.ndarray]]:
            """
            An [x, y, z] list of forces in wind axes [N]
            """
            return self.op_point.convert_axes(*self.F_g, from_axes="geometry", to_axes="wind")

        @property
        def M_b(self) -> List[Union[float, np.ndarray]]:
            """
            An [x, y, z] list of moments about body axes [Nm]
            """
            return self.op_point.convert_axes(*self.M_g, from_axes="geometry", to_axes="body")

        @property
        def M_w(self) -> List[Union[float, np.ndarray]]:
            """
            An [x, y, z] list of moments about wind axes [Nm]
            """
            return self.op_point.convert_axes(*self.M_g, from_axes="geometry", to_axes="wind")

        @property
        def L(self) -> Union[float, np.ndarray]:
            """
            The lift force [N]. Definitionally, this is in wind axes.
            """
            return -self.F_w[2]

        @property
        def Y(self) -> Union[float, np.ndarray]:
            """
            The side force [N]. Definitionally, this is in wind axes.
            """
            return self.F_w[1]

        @property
        def D(self) -> Union[float, np.ndarray]:
            """
            The drag force [N]. Definitionally, this is in wind axes.
            """
            return -self.F_w[0]

        @property
        def l_b(self) -> Union[float, np.ndarray]:
            """
            The rolling moment [Nm] in body axes. Positive is roll-right.
            """
            return self.M_b[0]

        @property
        def m_b(self) -> Union[float, np.ndarray]:
            """
            The pitching moment [Nm] in body axes. Positive is nose-up.
            """
            return self.M_b[1]

        @property
        def n_b(self) -> Union[float, np.ndarray]:
            """
            The yawing moment [Nm] in body axes. Positive is nose-right.
            """
            return self.M_b[2]

    def run(self) -> Dict:
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

        Nondimensional values are nondimensionalized using reference values in the LiftingLine.airplane object.

        Data types:
            - The "L", "Y", "D", "l_b", "m_b", "n_b", "CL", "CY", "CD", "Cl", "Cm", and "Cn" keys are:

                - floats if the OperatingPoint object is not vectorized (i.e., if all attributes of OperatingPoint
                are floats, not arrays).

                - arrays if the OperatingPoint object is vectorized (i.e., if any attribute of OperatingPoint is an
                array).

            - The "F_g", "F_b", "F_w", "M_g", "M_b", and "M_w" keys are always lists, which will contain either
            floats or arrays, again depending on whether the OperatingPoint object is vectorized or not.
        """
        aerobuildup = AeroBuildup(
            airplane=self.airplane,
            op_point=self.op_point,
            xyz_ref=self.xyz_ref,
        )

        wing_aero = self.wing_aerodynamics()

        fuselage_aero_components = [
            aerobuildup.fuselage_aerodynamics(
                fuselage=fuse,
                include_induced_drag=True,
            )
            for fuse in self.airplane.fuselages
        ]

        aero_components = [wing_aero] + fuselage_aero_components

        ### Sum up the forces
        F_g_total = [
            sum([comp.F_g[i] for comp in aero_components])
            for i in range(3)
        ]
        M_g_total = [
            sum([comp.M_g[i] for comp in aero_components])
            for i in range(3)
        ]

        ##### Start to assemble the output
        output = {
            "F_g": F_g_total,
            "M_g": M_g_total,
        }

        ##### Add in other metrics
        output["F_b"] = self.op_point.convert_axes(
            *F_g_total,
            from_axes="geometry",
            to_axes="body"
        )
        output["F_w"] = self.op_point.convert_axes(
            *F_g_total,
            from_axes="geometry",
            to_axes="wind"
        )
        output["M_b"] = self.op_point.convert_axes(
            *M_g_total,
            from_axes="geometry",
            to_axes="body"
        )
        output["M_w"] = self.op_point.convert_axes(
            *M_g_total,
            from_axes="geometry",
            to_axes="wind"
        )

        output["L"] = -output["F_w"][2]
        output["Y"] = output["F_w"][1]
        output["D"] = -output["F_w"][0]
        output["l_b"] = output["M_b"][0]
        output["m_b"] = output["M_b"][1]
        output["n_b"] = output["M_b"][2]

        ##### Compute dimensionalization factor
        qS = self.op_point.dynamic_pressure() * self.airplane.s_ref
        c = self.airplane.c_ref
        b = self.airplane.b_ref

        ##### Add nondimensional forces, and nondimensional quantities.
        output["CL"] = output["L"] / qS
        output["CY"] = output["Y"] / qS
        output["CD"] = output["D"] / qS
        output["Cl"] = output["l_b"] / qS / b
        output["Cm"] = output["m_b"] / qS / c
        output["Cn"] = output["n_b"] / qS / b

        ##### Add the component aerodynamics, for reference
        output["wing_aero"] = wing_aero
        output["fuselage_aero_components"] = fuselage_aero_components

        # ##### Add the drag breakdown
        # output["D_profile"] = sum([
        #     comp.D for comp in aero_components
        # ])
        # output["D_induced"] = D_induced

        return output

    def run_with_stability_derivatives(self,
                                       alpha=True,
                                       beta=True,
                                       p=True,
                                       q=True,
                                       r=True,
                                       ) -> Dict[str, Union[Union[float, np.ndarray], List[Union[float, np.ndarray]]]]:
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
            - 'L'   : the lift force [N]. Definitionally, this is in wind axes.
            - 'Y'   : the side force [N]. This is in wind axes.
            - 'D'   : the drag force [N]. Definitionally, this is in wind axes.
            - 'l_b' : the rolling moment, in body axes [Nm]. Positive is roll-right.
            - 'm_b' : the pitching moment, in body axes [Nm]. Positive is pitch-up.
            - 'n_b' : the yawing moment, in body axes [Nm]. Positive is nose-right.
            - 'CL'  : the lift coefficient [-]. Definitionally, this is in wind axes.
            - 'CY'  : the sideforce coefficient [-]. This is in wind axes.
            - 'CD'  : the drag coefficient [-]. Definitionally, this is in wind axes.
            - 'Cl'  : the rolling coefficient [-], in body axes
            - 'Cm'  : the pitching coefficient [-], in body axes
            - 'Cn'  : the yawing coefficient [-], in body axes

            Along with additional keys, depending on the value of the `alpha`, `beta`, `p`, `q`, and `r` arguments. For
            example, if `alpha=True`, then the following additional keys will be present:

                - 'CLa' : the lift coefficient derivative with respect to alpha [1/rad]
                - 'CDa' : the drag coefficient derivative with respect to alpha [1/rad]
                - 'CYa' : the sideforce coefficient derivative with respect to alpha [1/rad]
                - 'Cla' : the rolling moment coefficient derivative with respect to alpha [1/rad]
                - 'Cma' : the pitching moment coefficient derivative with respect to alpha [1/rad]
                - 'Cna' : the yawing moment coefficient derivative with respect to alpha [1/rad]
                - 'x_np': the neutral point location in the x direction [m]

            Nondimensional values are nondimensionalized using reference values in the AeroBuildup.airplane object.

            Data types:
                - The "L", "Y", "D", "l_b", "m_b", "n_b", "CL", "CY", "CD", "Cl", "Cm", and "Cn" keys are:

                    - floats if the OperatingPoint object is not vectorized (i.e., if all attributes of OperatingPoint
                    are floats, not arrays).

                    - arrays if the OperatingPoint object is vectorized (i.e., if any attribute of OperatingPoint is an
                    array).

                - The "F_g", "F_b", "F_w", "M_g", "M_b", and "M_w" keys are always lists, which will contain either
                floats or arrays, again depending on whether the OperatingPoint object is vectorized or not.

        """
        do_analysis: Dict[str, bool] = {
            "alpha": alpha,
            "beta" : beta,
            "p"    : p,
            "q"    : q,
            "r"    : r,
        }

        abbreviations: Dict[str, str] = {
            "alpha": "a",
            "beta" : "b",
            "p"    : "p",
            "q"    : "q",
            "r"    : "r",
        }
        finite_difference_amounts: Dict[str, float] = {
            "alpha": 0.001,
            "beta" : 0.001,
            "p"    : 0.001 * (2 * self.op_point.velocity) / self.airplane.b_ref,
            "q"    : 0.001 * (2 * self.op_point.velocity) / self.airplane.c_ref,
            "r"    : 0.001 * (2 * self.op_point.velocity) / self.airplane.b_ref,
        }
        scaling_factors: Dict[str, float] = {
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

        for d in do_analysis.keys():
            if not do_analysis[d]:  # Basically, if the parameter from the function input is not True,
                continue  # Skip this run.
                # This way, you can (optionally) speed up this routine if you only need static derivatives,
                # or longitudinal derivatives, etc.

            # These lines make a copy of the original operating point, incremented by the finite difference amount
            # along the variable defined by derivative_denominator.
            incremented_op_point = self.op_point.copy()
            if d == "alpha":
                incremented_op_point.alpha += finite_difference_amounts["alpha"]
            elif d == "beta":
                incremented_op_point.beta += finite_difference_amounts["beta"]
            elif d == "p":
                incremented_op_point.p += finite_difference_amounts["p"]
            elif d == "q":
                incremented_op_point.q += finite_difference_amounts["q"]
            elif d == "r":
                incremented_op_point.r += finite_difference_amounts["r"]
            else:
                raise ValueError(f"Invalid value of d: {d}!")

            aerobuildup_incremented = self.copy()
            aerobuildup_incremented.op_point = incremented_op_point
            run_incremented = aerobuildup_incremented.run()

            for derivative_numerator in [
                "CL",
                "CD",
                "CY",
                "Cl",
                "Cm",
                "Cn",
            ]:
                derivative_name = derivative_numerator + abbreviations[d]  # Gives "CLa"
                run_base[derivative_name] = (
                        (  # Finite-difference out the derivatives
                                run_incremented[derivative_numerator] - run_base[derivative_numerator]
                        ) / finite_difference_amounts[d]
                        * scaling_factors[d]
                )

            ### Try to compute and append neutral point, if possible
            if d == "alpha":
                Cma = run_base["Cma"]
                CLa = np.where(
                    run_base["CLa"] == 0,
                    np.nan,
                    run_base["CLa"],
                )

                run_base["x_np"] = self.xyz_ref[0] - (Cma / CLa * self.airplane.c_ref)

            if d == "beta":
                Cnb = run_base["Cnb"]
                CYb = np.where(
                    run_base["CYb"] == 0,
                    np.nan,
                    run_base["CYb"],
                )

                run_base["x_np_lateral"] = self.xyz_ref[0] - (Cnb / CYb * self.airplane.b_ref)

        return run_base

    def wing_aerodynamics(self) -> AeroComponentResults:

        if self.verbose:
            print("Meshing...")

        ##### Make Panels
        front_left_vertices = []
        back_left_vertices = []
        back_right_vertices = []
        front_right_vertices = []
        airfoils: List[Airfoil] = []
        control_surfaces: List[List[ControlSurface]] = []

        for wing in self.airplane.wings:  # subdivide the wing in more spanwise sections
            if self.spanwise_resolution > 1:
                wing = wing.subdivide_sections(
                    ratio=self.spanwise_resolution,
                    spacing_function=self.spanwise_spacing_function
                )

            points, faces = wing.mesh_thin_surface(
                method="quad",
                chordwise_resolution=1,
                add_camber=False,
            )

            front_left_vertices.append(points[faces[:, 0], :])
            back_left_vertices.append(points[faces[:, 1], :])
            back_right_vertices.append(points[faces[:, 2], :])
            front_right_vertices.append(points[faces[:, 3], :])

            wing_airfoils = []
            wing_control_surfaces = []

            for xsec_a, xsec_b in zip(wing.xsecs[:-1], wing.xsecs[1:]):  # Do the right side
                wing_airfoils.append(
                    xsec_a.airfoil.blend_with_another_airfoil(
                        airfoil=xsec_b.airfoil,
                        blend_fraction=0.5,
                    )
                )
                wing_control_surfaces.append(xsec_a.control_surfaces)

            airfoils.extend(wing_airfoils)
            control_surfaces.extend(wing_control_surfaces)

            if wing.symmetric:  # Do the left side, if applicable
                airfoils.extend(wing_airfoils)

                def mirror_control_surface(surf: ControlSurface) -> ControlSurface:
                    if surf.symmetric:
                        return surf
                    else:
                        surf = surf.copy()
                        surf.deflection *= -1
                        return surf

                symmetric_wing_control_surfaces = [
                    [mirror_control_surface(surf) for surf in surfs]
                    for surfs in wing_control_surfaces
                ]

                control_surfaces.extend(symmetric_wing_control_surfaces)

        front_left_vertices = np.concatenate(front_left_vertices)
        back_left_vertices = np.concatenate(back_left_vertices)
        back_right_vertices = np.concatenate(back_right_vertices)
        front_right_vertices = np.concatenate(front_right_vertices)

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
        vortex_bound_leg_norm = np.linalg.norm(vortex_bound_leg, axis=1)
        chord_vectors = (
                (back_left_vertices + back_right_vertices) / 2 -
                (front_left_vertices + front_right_vertices) / 2
        )
        chords = np.linalg.norm(chord_vectors, axis=1)
        wing_directions = vortex_bound_leg / tall(vortex_bound_leg_norm)
        local_forward_direction = np.cross(normal_directions, wing_directions)

        ### Save things to the instance for later access
        self.front_left_vertices = front_left_vertices
        self.back_left_vertices = back_left_vertices
        self.back_right_vertices = back_right_vertices
        self.front_right_vertices = front_right_vertices
        self.airfoils: List[Airfoil] = airfoils
        self.control_surfaces: List[List[ControlSurface]] = control_surfaces
        self.normal_directions = normal_directions
        self.areas = areas
        self.left_vortex_vertices = left_vortex_vertices
        self.right_vortex_vertices = right_vortex_vertices
        self.vortex_centers = vortex_centers
        self.vortex_bound_leg = vortex_bound_leg
        self.chord_vectors = chord_vectors
        self.chords = chords
        self.local_forward_direction = local_forward_direction
        self.n_panels = areas.shape[0]

        ##### Setup Operating Point
        if self.verbose:
            print("Calculating the freestream influence...")
        steady_freestream_velocity = self.op_point.compute_freestream_velocity_geometry_axes()  # Direction the wind is GOING TO, in geometry axes coordinates
        steady_freestream_direction = steady_freestream_velocity / np.linalg.norm(steady_freestream_velocity)

        steady_freestream_velocities = np.tile(
            wide(steady_freestream_velocity),
            reps=(self.n_panels, 1)
        )
        steady_freestream_directions = np.tile(
            wide(steady_freestream_direction),
            reps=(self.n_panels, 1)
        )

        rotation_freestream_velocities = self.op_point.compute_rotation_velocity_geometry_axes(
            points=vortex_centers
        )

        freestream_velocities = steady_freestream_velocities + rotation_freestream_velocities
        # Nx3, represents the freestream velocity at each panel collocation point (c)

        # freestream_influences = np.sum(freestream_velocities * normal_directions, axis=1)

        ### Save things to the instance for later access
        self.steady_freestream_velocity = steady_freestream_velocity
        self.steady_freestream_direction = steady_freestream_direction
        self.freestream_velocities = freestream_velocities

        ##### Compute the linearization quantities (CL0 and CLa) of each airfoil section
        alpha_geometrics = 90 - np.arccosd(
            np.sum(
                steady_freestream_directions * normal_directions,
                axis=1
            )
        )

        cos_sweeps = np.sum(
            steady_freestream_directions * -local_forward_direction,
            axis=1
        )

        machs = self.op_point.mach() * cos_sweeps

        Res = (
                      self.op_point.velocity *
                      chords /
                      self.op_point.atmosphere.kinematic_viscosity()
              ) * cos_sweeps

        # ### Do a central finite-difference in alpha to obtain CL0 and CLa quantities
        # finite_difference_alpha_amount = 1  # degree
        #
        # aero_finite_differences = [
        #     af.get_aero_from_neuralfoil(
        #         alpha=alpha_geometrics[i] + finite_difference_alpha_amount * np.array([-1, 1]),
        #         Re=Res[i],
        #         mach=machs[i],
        #         control_surfaces=control_surfaces[i]
        #     )
        #     for i, af in enumerate(airfoils)
        # ]
        #
        # CLs_at_alpha_geometric = [
        #     np.mean(aero["CL"])
        #     for aero in aero_finite_differences
        # ]
        # CLas = [
        #     np.diff(aero["CL"])[0] / (2 * np.radians(finite_difference_alpha_amount))
        #     for aero in aero_finite_differences
        # ]
        #
        # # Regularize CLas to always be positive and not-too-close-to-zero
        # CLas = [
        #     np.softmax(CLa, 1, softness=0.5)
        #     for CLa in CLas
        # ]

        ### OVERWRITE CL pre-calculation with more theory-based CLa (improves Cma prediction accuracy due to FD error)
        CLs_at_alpha_geometric = [
            af.get_aero_from_neuralfoil(
                alpha=alpha_geometrics[i],
                Re=Res[i],
                mach=machs[i],
                control_surfaces=control_surfaces[i],
                model_size=self.model_size,
            )["CL"]
            for i, af in enumerate(airfoils)
        ]

        CLas = 2 * np.pi * np.ones(len(CLs_at_alpha_geometric))

        ##### Setup Geometry
        ### Calculate AIC matrix
        if self.verbose:
            print("Calculating the collocation influence matrix...")

        u_centers_unit, v_centers_unit, w_centers_unit = calculate_induced_velocity_horseshoe(
            x_field=tall(vortex_centers[:, 0]),
            y_field=tall(vortex_centers[:, 1]),
            z_field=tall(vortex_centers[:, 2]),
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
                u_centers_unit * tall(normal_directions[:, 0]) +
                v_centers_unit * tall(normal_directions[:, 1]) +
                w_centers_unit * tall(normal_directions[:, 2])
        )
        # Influence of panel j's vortex strength [m^2/s] on panel i's normal-flow-velocity [m/s]

        alpha_influence_matrix = AIC / self.op_point.velocity
        # Influence of panel j's vortex strength [m^2/s] on panel i's alpha [radians]

        ##### Calculate Vortex Strengths
        if self.verbose:
            print("Calculating vortex center strengths (assembling and solving linear system)...")

        V_freestream_cross_li = np.cross(steady_freestream_velocities, self.vortex_bound_leg, axis=1)
        V_freestream_cross_li_magnitudes = np.linalg.norm(V_freestream_cross_li, axis=1)

        velocity_magnitude_perpendiculars = self.op_point.velocity * cos_sweeps

        A = alpha_influence_matrix * np.tile(wide(CLas), (self.n_panels, 1)) - np.diag(
            2 * V_freestream_cross_li_magnitudes / velocity_magnitude_perpendiculars ** 2 / areas
        )

        b = -1 * np.array(CLs_at_alpha_geometric)

        vortex_strengths = np.linalg.solve(A, b)
        self.vortex_strengths = vortex_strengths

        ##### Evaluate the aerodynamics with induced effects
        alpha_induced = np.degrees(alpha_influence_matrix @ vortex_strengths)

        alphas = tall(alpha_geometrics) + tall(alpha_induced)

        aeros = [
            af.get_aero_from_neuralfoil(
                alpha=alphas[i],
                Re=Res[i],
                mach=machs[i],
                control_surfaces=control_surfaces[i],
                model_size=self.model_size,
            )
            for i, af in enumerate(airfoils)
        ]
        CLs = np.array([aero["CL"] for aero in aeros])
        CDs = np.array([aero["CD"] for aero in aeros])
        CMs = np.array([aero["CM"] for aero in aeros])

        ##### Calculate forces
        ### Calculate Near-Field Forces and Moments
        # Governing Equation: The force on a straight, small vortex filament is F = rho * cross(V, l) * gamma,
        # where rho is density, V is the velocity vector, cross() is the cross product operator,
        # l is the vector of the filament itself, and gamma is the circulation.

        if self.verbose:
            print("Calculating induced forces on each panel...")
        # Calculate the induced velocity at the center of each bound leg

        velocities = self.get_velocity_at_points(
            points=vortex_centers,
            vortex_strengths=vortex_strengths
        )  # fuse added here

        velocity_magnitudes = np.linalg.norm(velocities, axis=1)

        # Calculate forces_inviscid_geometry, the force on the ith panel. Note that this is in GEOMETRY AXES,
        # not WIND AXES or BODY AXES.
        Vi_cross_li = np.cross(velocities, vortex_bound_leg, axis=1)

        forces_inviscid_geometry = self.op_point.atmosphere.density() * Vi_cross_li * tall(self.vortex_strengths)
        moments_inviscid_geometry = np.cross(
            np.add(vortex_centers, -wide(np.array(self.xyz_ref))),
            forces_inviscid_geometry
        )

        # Calculate total forces and moments
        force_inviscid_geometry = np.sum(forces_inviscid_geometry, axis=0)
        moment_inviscid_geometry = np.sum(moments_inviscid_geometry, axis=0)

        if self.verbose:
            print("Calculating profile forces and moments...")
        forces_profile_geometry = (
                0.5 * self.op_point.atmosphere.density() * velocities * tall(velocity_magnitudes)
                * tall(CDs) * tall(areas)
        )

        moments_profile_geometry = np.cross(
            np.add(vortex_centers, -wide(np.array(self.xyz_ref))),
            forces_profile_geometry
        )
        force_profile_geometry = np.sum(forces_profile_geometry, axis=0)
        moment_profile_geometry = np.sum(moments_profile_geometry, axis=0)

        # Compute pitching moment

        bound_leg_YZ = vortex_bound_leg
        bound_leg_YZ[:, 0] = 0
        moments_pitching_geometry = (0.5 * self.op_point.atmosphere.density() * tall(velocity_magnitudes ** 2)) \
                                    * tall(CMs) * tall(chords ** 2) * bound_leg_YZ
        moment_pitching_geometry = np.sum(moments_pitching_geometry, axis=0)

        if self.verbose:
            print("Calculating total forces and moments...")

        force_total_geometry = np.add(force_inviscid_geometry, force_profile_geometry)
        moment_total_geometry = np.add(moment_inviscid_geometry, moment_profile_geometry) + moment_pitching_geometry

        return self.AeroComponentResults(
            s_ref=self.airplane.s_ref,
            c_ref=self.airplane.c_ref,
            b_ref=self.airplane.b_ref,
            op_point=self.op_point,
            F_g=force_total_geometry,
            M_g=moment_total_geometry,
        )

    def get_induced_velocity_at_points(self,
                                       points: np.ndarray,
                                       vortex_strengths: np.ndarray = None
                                       ) -> np.ndarray:
        """
        Computes the induced velocity at a set of points in the flowfield.

        Args:
            points: A Nx3 array of points that you would like to know the induced velocities at. Given in geometry axes.

        Returns: A Nx3 of the induced velocity at those points. Given in geometry axes.

        """
        if vortex_strengths is None:
            try:
                vortex_strengths = self.vortex_strengths
            except AttributeError:
                raise ValueError(
                    "`LiftingLine.vortex_strengths` doesn't exist, so you need to pass in the `vortex_strengths` parameter.")

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
            trailing_vortex_direction=(
                self.steady_freestream_direction
                if self.align_trailing_vortices_with_wind else
                np.array([1, 0, 0])
            ),
            gamma=wide(vortex_strengths),
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
                               points: np.ndarray,
                               vortex_strengths: np.ndarray = None,
                               ) -> np.ndarray:
        """
        Computes the velocity at a set of points in the flowfield.

        Args:
            points: A Nx3 array of points that you would like to know the velocities at. Given in geometry axes.

        Returns: A Nx3 of the velocity at those points. Given in geometry axes.

        """
        V_induced = self.get_induced_velocity_at_points(
            points=points,
            vortex_strengths=vortex_strengths,
        )

        rotation_freestream_velocities = np.array(self.op_point.compute_rotation_velocity_geometry_axes(
            points
        ))

        freestream_velocities = np.add(wide(self.steady_freestream_velocity), rotation_freestream_velocities)

        V = V_induced + freestream_velocities

        # if self.airplane.fuselages:
        #     V_induced_fuselage = self.calculate_fuselage_influences(
        #         points=points
        #     )
        #     V = V + V_induced_fuselage

        return V

    def calculate_fuselage_influences(self, points: np.ndarray) -> np.ndarray:

        this_fuse_centerline_points = []  # fuselage sections centres
        this_fuse_radii = []

        for fuse in self.airplane.fuselages:  # iterating through the airplane fuselages
            for xsec_num in range(len(fuse.xsecs)):  # iterating through the current fuselage sections
                this_fuse_xsec = fuse.xsecs[xsec_num]
                this_fuse_centerline_points.append(this_fuse_xsec.xyz_c)
                this_fuse_radii.append(this_fuse_xsec.width / 2)

        this_fuse_centerline_points = np.stack(
            this_fuse_centerline_points,
            axis=0
        )
        this_fuse_centerline_points = (this_fuse_centerline_points[1:, :] +
                                       this_fuse_centerline_points[:-1, :]) / 2
        this_fuse_radii = np.array(this_fuse_radii)

        areas = np.pi * this_fuse_radii ** 2
        freestream_x_component = self.op_point.compute_freestream_velocity_geometry_axes()[
            0]  # TODO add in rotation corrections, add in doublets for alpha
        sigmas = freestream_x_component * np.diff(areas)

        u_induced_fuse, v_induced_fuse, w_induced_fuse = calculate_induced_velocity_point_source(
            x_field=tall(points[:, 0]),
            y_field=tall(points[:, 1]),
            z_field=tall(points[:, 2]),
            x_source=wide(this_fuse_centerline_points[:, 0]),
            y_source=wide(this_fuse_centerline_points[:, 1]),
            z_source=wide(this_fuse_centerline_points[:, 2]),
            sigma=wide(sigmas),
            viscous_radius=0.0001,
        )

        # # Compressibility
        # dy *= self.beta
        # dz *= self.beta

        # For now, we're just putting a point source at the middle... # TODO make an actual line source
        # source_x = (dx[:, 1:] + dx[:, :-1]) / 2
        # source_y = (dy[:, 1:] + dy[:, :-1]) / 2
        # source_z = (dz[:, 1:] + dz[:, :-1]) / 2

        fuselage_influences_x = np.sum(u_induced_fuse, axis=1)
        fuselage_influences_y = np.sum(v_induced_fuse, axis=1)
        fuselage_influences_z = np.sum(w_induced_fuse, axis=1)

        fuselage_influences = np.stack([
            fuselage_influences_x, fuselage_influences_y, fuselage_influences_z
        ], axis=1)

        return fuselage_influences

    def calculate_streamlines(self,
                              seed_points: np.ndarray = None,
                              n_steps: int = 300,
                              length: float = None
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
            left_TE_vertices = self.back_left_vertices
            right_TE_vertices = self.back_right_vertices
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
                colorbar_title=colorbar_label
                               ** show_kwargs,
            )

        elif backend == "pyvista":
            import pyvista as pv
            plotter = pv.Plotter()
            plotter.title = "ASB LiftingLine"
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
            # # Fuselages
            # for fuse_id in range(len(self.airplane.fuselages)):
            #     fuse = self.airplane.fuselages[fuse_id]  # type: Fuselage
            #
            #     for xsec_id in range(len(fuse.xsecs) - 1):
            #         xsec_1 = fuse.xsecs[xsec_id]  # type: FuselageXSec
            #         xsec_2 = fuse.xsecs[xsec_id + 1]  # type: FuselageXSec
            #
            #         r1 = xsec_1.equivalent_radius(preserve="area")
            #         r2 = xsec_2.equivalent_radius(preserve="area")
            #         points_1 = np.zeros((fuse.xsec_perimeter, 3))
            #         points_2 = np.zeros((fuse.xsec_perimeter, 3))
            #         for point_index in range(fuse.xsec_perimeter):
            #             from aerosandbox.numpy import rotation_matrix_3D
            #             rot = rotation_matrix_3D(
            #                 2 * np.pi * point_index / fuse.xsec_perimeter,
            #                 [1, 0, 0],
            #                 True
            #             ).toarray()
            #             points_1[point_index, :] = rot @ np.array([0, 0, r1])
            #             points_2[point_index, :] = rot @ np.array([0, 0, r2])
            #         points_1 = points_1 + np.array(xsec_1.xyz_c).reshape(-1)
            #         points_2 = points_2 + np.array(xsec_2.xyz_c).reshape(-1)
            #
            #         for point_index in range(fuse.circumferential_panels):
            #
            #             fig.add_quad(points=[
            #                 points_1[(point_index) % fuse.xsec_perimeter, :],
            #                 points_1[(point_index + 1) % fuse.xsec_perimeter, :],
            #                 points_2[(point_index + 1) % fuse.xsec_perimeter, :],
            #                 points_2[(point_index) % fuse.xsec_perimeter, :],
            #             ],
            #                 intensity=0,
            #             )

            # if draw_streamlines:
            #     if (not hasattr(self, 'streamlines')) or recalculate_streamlines:
            #         if self.verbose:
            #             print("Calculating streamlines...")
            #         seed_points = (back_left_vertices + back_right_vertices) / 2
            #         self.calculate_streamlines(seed_points=seed_points)
            #
            #     if self.verbose:
            #         print("Parsing streamline data...")
            #     n_streamlines = self.streamlines[0].shape[0]
            #     n_timesteps = len(self.streamlines)
            #
            #     for streamlines_num in range(n_streamlines):
            #         streamline = [self.streamlines[ts][streamlines_num, :] for ts in range(n_timesteps)]
            #         fig.add_streamline(
            #             points=streamline,
            #             mirror=self.run_symmetric
            #         )


if __name__ == '__main__':
    import aerosandbox as asb
    import aerosandbox.numpy as np
    from aerosandbox.aerodynamics.aero_3D.test_aero_3D.geometries.conventional import airplane
    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    airplane = asb.Airplane(
        wings=[
            asb.Wing(
                symmetric=False,
                xsecs=[
                    asb.WingXSec(
                        xyz_le=[0, 0, 0],
                        chord=1,
                        airfoil=asb.Airfoil("naca0012")
                    ),
                    asb.WingXSec(
                        xyz_le=[0, 5, 0],
                        chord=1,
                        airfoil=asb.Airfoil("naca0012")
                    ),
                ]
            )
        ]
    )

    ll = LiftingLine(
        airplane=airplane,
        op_point=OperatingPoint(
            velocity=100,
            alpha=5,
            beta=0,
        ),
        xyz_ref=[0.25, 0, 0],
        spanwise_resolution=5,
    )

    aero = ll.run()

    print(aero["CL"])
    print(aero["CL"] / aero["CD"])


    @np.vectorize
    def get_aero(resolution):
        return LiftingLine(
            airplane=airplane,
            op_point=OperatingPoint(
                velocity=100,
                alpha=5,
                beta=0,
            ),
            xyz_ref=[0.25, 0, 0],
            spanwise_resolution=resolution,
        ).run()

    resolutions = 1 + np.arange(20)
    aeros = get_aero(resolutions)

    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    fig, ax = plt.subplots(3, 1)
    ax[0].semilogx(
        resolutions,
        [aero["CL"] for aero in aeros]
    )
    ax[0].plot(
        resolutions,
        np.array([aero["CL"] for aero in aeros]) * (resolutions) / (resolutions + 0.3),
        # np.array([aero["CL"] for aero in aeros]) * (resolutions) / (resolutions + 0.3),
    )
    ax[1].plot(
        resolutions,
        [aero["CD"] for aero in aeros]
    )
    ax[1].plot(
        resolutions,
        np.array([aero["CD"] for aero in aeros]) * (resolutions + 0.2) / (resolutions),
    )
    ax[2].plot(
        resolutions,
        [aero["CL"] / aero["CD"] for aero in aeros]
    )
    p.show_plot()
