from aerosandbox import ImplicitAnalysis
from aerosandbox.geometry import *
from aerosandbox.performance import OperatingPoint
from aerosandbox.aerodynamics.aero_3D.singularities.uniform_strength_horseshoe_singularities import \
    calculate_induced_velocity_horseshoe
from aerosandbox.aerodynamics.aero_3D.singularities.point_source import \
    calculate_induced_velocity_point_source
import aerosandbox.numpy as np
from typing import Dict, Any, Callable, List
import copy


### Define some helper functions that take a vector and make it a Nx1 or 1xN, respectively.
# Useful for broadcasting with matrices later.
def tall(array):
    return np.reshape(array, (-1, 1))


def wide(array):
    return np.reshape(array, (1, -1))


class NonlinearLiftingLine(ImplicitAnalysis):
    """
    An implicit aerodynamics analysis based on lifting line theory, with modifications for nonzero sweep
    and dihedral + multiple wings.

    Nonlinear, and includes viscous effects based on 2D data.

    Usage example:
        >>> analysis = asb.NonlinearLiftingLine(
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

    @ImplicitAnalysis.initialize
    def __init__(self,
                 airplane: Airplane,
                 op_point: OperatingPoint,
                 xyz_ref: List[float] = None,
                 run_symmetric_if_possible: bool = False,
                 verbose: bool = False,
                 spanwise_resolution=8,  # TODO document
                 spanwise_spacing_function: Callable[[float, float, float], np.ndarray] = np.cosspace,
                 vortex_core_radius: float = 1e-8,
                 align_trailing_vortices_with_wind: bool = False,
                 ):
        """
        Initializes and conducts a NonlinearLiftingLine analysis.

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

        ### Initialize

        if xyz_ref is None:
            xyz_ref = airplane.xyz_ref

        self.airplane = airplane
        self.op_point = op_point
        self.xyz_ref = xyz_ref
        self.verbose = verbose
        self.spanwise_resolution = spanwise_resolution
        self.spanwise_spacing_function = spanwise_spacing_function
        self.vortex_core_radius = vortex_core_radius
        self.align_trailing_vortices_with_wind = align_trailing_vortices_with_wind

        ### Determine whether you should run the problem as symmetric
        self.run_symmetric = False
        if run_symmetric_if_possible:
            raise NotImplementedError("LL with symmetry detection not yet implemented!")
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

    def run(self, solve: bool = True) -> Dict[str, Any]:
        """
        Computes the aerodynamic forces.

        Returns a dictionary with keys:
            - 'residuals': a list of residuals for each horseshoe element
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
            - 'CDi' the induced drag coefficient
            - 'CDp' the profile drag coefficient
            - 'Cl', the rolling coefficient [-], in body axes
            - 'Cm', the pitching coefficient [-], in body axes
            - 'Cn', the yawing coefficient [-], in body axes

        Nondimensional values are nondimensionalized using reference values in the VortexLatticeMethod.airplane object.
        """

        self.solve = solve  # is it is True (default), NL_lifting_line is a standalone solver. If False, it
        # is expected to constrain the residuals to zero in the outer optimization problem

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

        ##### Setup Operating Point
        if self.verbose:
            print("Calculating the freestream influence...")
        steady_freestream_velocity = self.op_point.compute_freestream_velocity_geometry_axes()  # Direction the wind is GOING TO, in geometry axes coordinates
        steady_freestream_direction = steady_freestream_velocity / np.linalg.norm(steady_freestream_velocity)
        rotation_freestream_velocities = self.op_point.compute_rotation_velocity_geometry_axes(
            vortex_centers)

        freestream_velocities = np.add(wide(steady_freestream_velocity), rotation_freestream_velocities)
        # Nx3, represents the freestream velocity at each panel collocation point (c)

        freestream_influences = np.sum(freestream_velocities * normal_directions, axis=1)

        ### Save things to the instance for later access
        self.steady_freestream_velocity = steady_freestream_velocity
        self.steady_freestream_direction = steady_freestream_direction
        self.freestream_velocities = freestream_velocities

        ##### Calculate Vortex Strengths
        if self.verbose:
            print("Calculating vortex center strengths...")

        self.n_panels = areas.shape[0]

        # Set up implicit solve (explicit is not possible for general nonlinear problem)
        vortex_strengths = self.opti.variable(init_guess=np.zeros(shape=self.n_panels))
        # scale =self.op_point.velocity) )
        self.vortex_strengths = vortex_strengths

        # Find velocities
        velocities = self.get_velocity_at_points(
            points=self.vortex_centers,
            vortex_strengths=vortex_strengths
        )  # TODO just a reminder, fuse added here

        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        velocity_directions = velocities / tall(velocity_magnitudes)

        alphas = 90 - np.arccosd(
            np.sum(velocity_directions * self.normal_directions, axis=1)
        )

        # Get perpendicular parameters
        cos_sweeps = np.sum(velocity_directions * -local_forward_direction, axis=1)

        Res = (
                      velocity_magnitudes *
                      self.chords /
                      self.op_point.atmosphere.kinematic_viscosity()
              ) * cos_sweeps

        machs = velocity_magnitudes / self.op_point.atmosphere.speed_of_sound() * cos_sweeps

        aeros = [
            af.get_aero_from_neuralfoil(
                alpha=alphas[i],
                Re=Res[i],
                mach=machs[i],
                control_surfaces=self.control_surfaces[i],
            )
            for i, af in enumerate(self.airfoils)
        ]
        CLs = np.array([aero["CL"] for aero in aeros])
        CDs = np.array([aero["CD"] for aero in aeros])
        CMs = np.array([aero["CM"] for aero in aeros])

        self.CLs = CLs
        self.CDs = CDs
        self.CMs = CMs
        self.alphas = alphas

        Vi_cross_li = np.cross(velocities, self.vortex_bound_leg, axis=1)
        Vi_cross_li_magnitudes = np.linalg.norm(Vi_cross_li, axis=1)

        # self.opti.subject_to([
        #     vortex_strengths * Vi_cross_li_magnitudes ==
        #     0.5 * velocity_magnitude_perpendiculars ** 2 * self.CLs * self.areas
        # ])
        velocity_magnitude_perpendiculars = velocity_magnitudes * cos_sweeps

        residuals = (
                vortex_strengths * Vi_cross_li_magnitudes * 2 / velocity_magnitude_perpendiculars ** 2 / areas - CLs
        )

        if self.solve:
            self.opti.subject_to([
                residuals == 0
            ])

            self.sol = self.opti.solve(verbose=False)
            self.vortex_strengths = self.sol(vortex_strengths)

        ##### Calculate forces
        ### Calculate Near-Field Forces and Moments
        # Governing Equation: The force on a straight, small vortex filament is F = rho * cross(V, l) * gamma,
        # where rho is density, V is the velocity vector, cross() is the cross product operator,
        # l is the vector of the filament itself, and gamma is the circulation.

        if self.verbose:
            print("Calculating induced forces on each panel...")
        # Calculate the induced velocity at the center of each bound leg

        velocities = self.get_velocity_at_points(
            points=self.vortex_centers,
            vortex_strengths=self.vortex_strengths
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

        if self.solve:
            CDs = self.sol(CDs)
            CMs = self.sol(CMs)
            residuals = self.sol(residuals)

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

        # # Inviscid force from geometry to body and wind axes
        force_inviscid_body = np.array(self.op_point.convert_axes(
            force_inviscid_geometry[0], force_inviscid_geometry[1], force_inviscid_geometry[2],
            from_axes="geometry",
            to_axes="body"
        ))
        force_inviscid_wind = np.array(self.op_point.convert_axes(
            force_inviscid_body[0], force_inviscid_body[1], force_inviscid_body[2],
            from_axes="body",
            to_axes="wind"
        ))

        # # Profile force from geometry to body and wind axes
        force_profile_body = np.array(self.op_point.convert_axes(
            force_profile_geometry[0], force_profile_geometry[1], force_profile_geometry[2],
            from_axes="geometry",
            to_axes="body"
        ))
        force_profile_wind = np.array(self.op_point.convert_axes(
            force_profile_body[0], force_profile_body[1], force_profile_body[2],
            from_axes="body",
            to_axes="wind"
        ))

        # Compute pitching moment

        bound_leg_YZ = vortex_bound_leg
        bound_leg_YZ[:, 0] = 0
        moments_pitching_geometry = (0.5 * self.op_point.atmosphere.density() * tall(velocity_magnitudes ** 2)) \
                                    * tall(CMs) * tall(chords ** 2) * bound_leg_YZ
        moment_pitching_geometry = np.sum(moments_pitching_geometry, axis=0)

        if self.verbose:
            print("Calculating total forces and moments...")
        force_total_geometry = np.add(force_inviscid_geometry, force_profile_geometry)

        force_total_body = np.array(self.op_point.convert_axes(
            force_total_geometry[0], force_total_geometry[1], force_total_geometry[2],
            from_axes="geometry",
            to_axes="body"
        ))
        force_total_wind = np.array(self.op_point.convert_axes(
            force_total_body[0], force_total_body[1], force_total_body[2],
            from_axes="body",
            to_axes="wind"
        ))

        moment_total_geometry = np.add(moment_inviscid_geometry, moment_profile_geometry) + moment_pitching_geometry

        moment_total_body = np.array(self.op_point.convert_axes(
            moment_total_geometry[0], moment_total_geometry[1], moment_total_geometry[2],
            from_axes="geometry",
            to_axes="body"
        ))
        moment_total_wind = np.array(self.op_point.convert_axes(
            moment_total_body[0], moment_total_body[1], moment_total_body[2],
            from_axes="body",
            to_axes="wind"
        ))

        ### Save things to the instance for later access
        L = -force_total_wind[2]
        D = -force_total_wind[0]
        Di = -force_inviscid_wind[0]
        Dp = -force_profile_wind[0]
        Y = force_total_wind[1]
        l_b = moment_total_body[0]
        m_b = moment_total_body[1]
        n_b = moment_total_body[2]

        # Calculate nondimensional forces
        q = self.op_point.dynamic_pressure()
        s_ref = self.airplane.s_ref
        b_ref = self.airplane.b_ref
        c_ref = self.airplane.c_ref
        self.CL = L / q / s_ref
        self.CD = D / q / s_ref
        CDi = Di / q / s_ref
        CDp = Dp / q / s_ref
        self.CY = Y / q / s_ref
        self.Cl = l_b / q / s_ref / b_ref
        self.Cm = m_b / q / s_ref / c_ref
        self.Cn = n_b / q / s_ref / b_ref

        self.CL_over_CD = np.where(self.CD == 0, 0, np.array(self.CL / self.CD))

        return {
            "residuals" : residuals,
            "F_g"       : force_total_geometry,
            "F_b"       : force_total_body,
            "F_w"       : force_total_wind,
            "M_g"       : moment_total_geometry,
            "M_b"       : moment_total_body,
            "M_w"       : moment_total_wind,
            "L"         : L,
            "D"         : D,
            "Y"         : Y,
            "l_b"       : l_b,
            "m_b"       : m_b,
            "n_b"       : n_b,
            "CL"        : self.CL,
            "CD"        : self.CD,
            "CDi"       : CDi,
            "CDp"       : CDp,
            "CY"        : self.CY,
            "CL_over_CD": self.CL_over_CD,
            "Cl"        : self.Cl,
            "Cm"        : self.Cm,
            "Cn"        : self.Cn
        }

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
                    "`NonlinearLiftingLine.vortex_strengths` doesn't exist, so you need to pass in the `vortex_strengths` parameter.")

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
            plotter.title = "ASB NonlinearLiftingLine"
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
    ### Import Vanilla Airplane
    import aerosandbox as asb

    from pathlib import Path

    geometry_folder = Path(__file__).parent / "test_aero_3D" / "geometries"

    import sys

    sys.path.insert(0, str(geometry_folder))

    from UniqueWing import airplane as vanilla

    ### Do the AVL run
    LL_aeros = NonlinearLiftingLine(
        airplane=vanilla,
        op_point=asb.OperatingPoint(
            atmosphere=asb.Atmosphere(altitude=0),
            velocity=10,  # m/s
            alpha=5),
        verbose=True,
        spanwise_resolution=5,
    )

    res = LL_aeros.run()

    for k, v in res.items():
        print(f"{str(k).rjust(10)} : {v}")

    # LL_aeros.draw(
    #     c=LL_aeros.CL
    # )
