from aerosandbox import Opti, ImplicitAnalysis
from aerosandbox.geometry import *
from aerosandbox.performance import OperatingPoint
from aerosandbox.aerodynamics.aero_3D.singularities.uniform_strength_horseshoe_singularities import \
    calculate_induced_velocity_horseshoe
from typing import Dict


### Define some helper functions that take a vector and make it a Nx1 or 1xN, respectively.
# Useful for broadcasting with matrices later.
def tall(array):
    return np.reshape(array, (-1, 1))


def wide(array):
    return np.reshape(array, (1, -1))


class LiftingLine(ImplicitAnalysis):
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
                 run_symmetric_if_possible=True,
                 verbose=True,
                 spanwise_resolution=8,  # TODO document
                 spanwise_spacing="cosine",  # TODO document,
                 vortex_core_radius: float = 1e-8,
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

        ### Initialize
        self.airplane = airplane
        self.op_point = op_point
        self.run_symmetric_if_possible = run_symmetric_if_possible
        self.verbose = verbose
        self.spanwise_resolution = spanwise_resolution
        self.spanwise_spacing = spanwise_spacing
        self.vortex_core_radius = vortex_core_radius

        ### Determine whether you should run the problem as symmetric
        self.run_symmetric = False
        if self.run_symmetric_if_possible:
            try:
                self.run_symmetric = (  # Satisfies assumptions
                        self.op_point.beta == 0 and
                        self.op_point.p == 0 and
                        self.op_point.r == 0 and
                        self.airplane.is_entirely_symmetric()
                )
            except RuntimeError:  # Required because beta, p, r, etc. may be non-numeric (e.g. opti variables)
                pass

    def run(self) -> Dict:
        self.setup_mesh()

    def setup_mesh(self) -> None:
        if self.verbose:
            print("Meshing...")

        ##### Make Panels
        front_left_vertices = []
        back_left_vertices = []
        back_right_vertices = []
        front_right_vertices = []
        CL_functions = []
        CD_functions = []
        CM_functions = []
        wing_ids = []

        for wing_id, wing in enumerate(self.airplane.wings):  # Iterate through wings
            for xsec_a, xsec_b in zip(
                    wing.xsecs[:-1],
                    wing.xsecs[1:]
            ):  # Iterate through pairs of wing cross sections
                wing_section = Wing(
                    xsecs=[
                        xsec_a,  # Inside cross section
                        xsec_b  # Outside cross section
                    ],
                    symmetric=wing.symmetric
                )

                points, faces = wing_section.mesh_thin_surface(
                    method="quad",
                    chordwise_resolution=1,
                    spanwise_resolution=self.spanwise_resolution,
                    spanwise_spacing=self.spanwise_spacing,
                    add_camber=False
                )
                front_left_vertices.append(points[faces[:, 0], :])
                back_left_vertices.append(points[faces[:, 1], :])
                back_right_vertices.append(points[faces[:, 2], :])
                front_right_vertices.append(points[faces[:, 3], :])
                wing_ids.append(wing_id * np.ones(len(faces)))

                if self.spanwise_spacing == 'uniform':
                    y_nondim_vertices = np.linspace(0, 1, self.spanwise_resolution + 1)
                elif self.spanwise_spacing == 'cosine':
                    y_nondim_vertices = np.cosspace(0, 1, self.spanwise_resolution + 1)
                else:
                    raise Exception("Bad value of `LiftingLine.spanwise_spacing`!")

                y_nondim = (y_nondim_vertices[:-1] + y_nondim_vertices[1:]) / 2
                if wing_section.symmetric:
                    y_nondim = np.concatenate([y_nondim, y_nondim])

                for y_nondim_i in y_nondim:
                    CL_functions.append(
                        lambda alpha, Re, mach,
                               xsec_a=xsec_a, xsec_b=xsec_b, y_nondim=y_nondim_i:
                        xsec_a.airfoil.CL_function(alpha, Re, mach) * (1 - y_nondim) +
                        xsec_b.airfoil.CL_function(alpha, Re, mach) * (y_nondim)
                    )
                    CD_functions.append(
                        lambda alpha, Re, mach,
                               xsec_a=xsec_a, xsec_b=xsec_b, y_nondim=y_nondim_i:
                        xsec_a.airfoil.CD_function(alpha, Re, mach) * (1 - y_nondim) +
                        xsec_b.airfoil.CD_function(alpha, Re, mach) * (y_nondim)
                    )
                    CM_functions.append(
                        lambda alpha, Re, mach,
                               xsec_a=xsec_a, xsec_b=xsec_b, y_nondim=y_nondim_i:
                        xsec_a.airfoil.CM_function(alpha, Re, mach) * (1 - y_nondim) +
                        xsec_b.airfoil.CM_function(alpha, Re, mach) * (y_nondim)
                    )

        front_left_vertices = np.concatenate(front_left_vertices)
        back_left_vertices = np.concatenate(back_left_vertices)
        back_right_vertices = np.concatenate(back_right_vertices)
        front_right_vertices = np.concatenate(front_right_vertices)
        wing_ids = np.concatenate(wing_ids)

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
        chord_vectors = (
                (back_left_vertices + back_right_vertices) / 2 -
                (front_left_vertices + front_right_vertices) / 2
        )
        chords = np.linalg.norm(chord_vectors, axis=1)

        ### Save things to the instance for later access
        self.front_left_vertices = front_left_vertices
        self.back_left_vertices = back_left_vertices
        self.back_right_vertices = back_right_vertices
        self.front_right_vertices = front_right_vertices
        self.CL_functions = CL_functions  # type: list # of callables
        self.CD_functions = CD_functions  # type: list # of callables
        self.CM_functions = CM_functions  # type: list # of callables
        self.wing_ids = wing_ids
        self.normal_directions = normal_directions
        self.areas = areas
        self.left_vortex_vertices = left_vortex_vertices
        self.right_vortex_vertices = right_vortex_vertices
        self.vortex_centers = vortex_centers
        self.vortex_bound_leg = vortex_bound_leg
        self.chord_vectors = chord_vectors
        self.chords = chords

        if self.verbose:
            print("Calculating the freestream influence...")
        steady_freestream_velocity = self.op_point.compute_freestream_velocity_geometry_axes()  # Direction the wind is GOING TO, in geometry axes coordinates
        steady_freestream_direction = steady_freestream_velocity / np.linalg.norm(steady_freestream_velocity)
        rotation_freestream_velocities = self.op_point.compute_rotation_velocity_geometry_axes(
            vortex_centers)

        freestream_velocities = wide(steady_freestream_velocity) + rotation_freestream_velocities
        # Nx3, represents the freestream velocity at each panel collocation point (c)

        freestream_influences = np.sum(freestream_velocities * normal_directions, axis=1)

        ### Save things to the instance for later access
        self.steady_freestream_velocity = steady_freestream_velocity
        self.steady_freestream_direction = steady_freestream_direction
        self.freestream_velocities = freestream_velocities

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
            trailing_vortex_direction=self.steady_freestream_direction,
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

        rotation_freestream_velocities = self.op_point.compute_rotation_velocity_geometry_axes(
            points
        )

        freestream_velocities = self.steady_freestream_velocity + rotation_freestream_velocities

        V = V_induced + freestream_velocities
        return V

    def compute_solution_quantities(self, vortex_strengths: np.ndarray) -> Dict:
        velocities = self.get_velocity_at_points(
            points=self.vortex_centers,
            vortex_strengths=vortex_strengths
        )
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        velocity_directions = velocities / tall(velocity_magnitudes)

        alphas = 90 - np.arccosd(
            np.sum(velocity_directions * self.normal_directions, axis=1)
        )

        Res = (
                velocity_magnitudes *
                self.chords /
                self.op_point.atmosphere.kinematic_viscosity()
        )  # TODO add multiply by cos_sweeps
        machs = velocity_magnitudes / self.op_point.atmosphere.speed_of_sound()

        CLs, CDs, CMs = [
            np.array([
                polar_function(
                    alpha=alphas[i],
                    Re=Res[i],
                    mach=machs[i],
                )
                for i, polar_function in enumerate(polar_functions)
            ])
            for polar_functions in [
                self.CL_functions,
                self.CD_functions,
                self.CM_functions
            ]
        ]

        Vi_cross_li = np.cross(velocities, self.vortex_bound_leg, axis=1)
        Vi_cross_li_magnitudes = np.linalg.norm(Vi_cross_li, axis=1)

        residuals = (
                vortex_strengths * Vi_cross_li_magnitudes * 2 / velocity_magnitudes ** 2 / self.areas - CLs
        )

        return {
            "residuals": residuals,
            "alphas"   : alphas,
            "Res"      : Res,
            "machs"    : machs,
            "CLs"      : CLs,
            "CDs"      : CDs,
            "CMs"      : CMs,
        }

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
            left_TE_vertices = self.back_left_vertices[self.is_trailing_edge]
            right_TE_vertices = self.back_right_vertices[self.is_trailing_edge]
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
        To solve an AeroProblem, use opti.solve(). To substitute a solved solution, use ap = ap.substitute_solution(sol).
        :return:
        """

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
                plotter.show()
            return plotter




        else:
            raise ValueError("Bad value of `backend`!")

    def _setup_geometry(self):
        if self.verbose:
            print("Calculating the vortex center velocity influence matrix...")
        self.Vij_x, self.Vij_y, self.Vij_z = self.calculate_Vij(self.vortex_centers)

        if self.verbose:
            print("Calculating fuselage influences...")
        self.beta = np.sqrt(1 - self.op_point.mach())
        self.fuselage_velocities = self.calculate_fuselage_influences(self.vortex_centers)
        # TODO do this

    def _setup_operating_point(self):
        if self.verbose:
            print("Calculating the freestream influence...")
        self.steady_freestream_velocity = self.op_point.compute_freestream_velocity_geometry_axes()  # Direction the wind is GOING TO, in geometry axes coordinates
        self.rotation_freestream_velocities = self.op_point.compute_rotation_velocity_geometry_axes(
            self.vortex_centers)
        self.freestream_velocities = cas.transpose(self.steady_freestream_velocity + cas.transpose(
            self.rotation_freestream_velocities))  # Nx3, represents the freestream velocity at each vortex center

    def _calculate_vortex_strengths(self):
        if self.verbose:
            print("Calculating vortex strengths...")

        # Set up implicit solve (explicit is not possible for general nonlinear problem)
        self.vortex_strengths = self.opti.variable(self.n_panels)
        self.opti.set_initial(self.vortex_strengths, 0)

        # Find velocities
        self.induced_velocities = cas.horzcat(
            self.Vij_x @ self.vortex_strengths,
            self.Vij_y @ self.vortex_strengths,
            self.Vij_z @ self.vortex_strengths,
        )
        self.velocities = self.induced_velocities + self.freestream_velocities + self.fuselage_velocities  # TODO just a reminder, fuse added here
        self.alpha_eff_perpendiculars = cas.atan2(
            (
                    self.velocities[:, 0] * self.normal_directions[:, 0] +
                    self.velocities[:, 1] * self.normal_directions[:, 1] +
                    self.velocities[:, 2] * self.normal_directions[:, 2]
            ),
            (
                    self.velocities[:, 0] * -self.local_forward_directions[:, 0] +
                    self.velocities[:, 1] * -self.local_forward_directions[:, 1] +
                    self.velocities[:, 2] * -self.local_forward_directions[:, 2]
            )
        ) * (180 / cas.pi)
        self.velocity_magnitudes = np.sqrt(
            self.velocities[:, 0] ** 2 +
            self.velocities[:, 1] ** 2 +
            self.velocities[:, 2] ** 2
        )
        self.Res = self.op_point.density * self.velocity_magnitudes * self.chords / self.op_point.viscosity
        self.machs = [self.op_point.mach] * self.n_panels  # TODO incorporate sweep effects here!

        # Get perpendicular parameters
        self.cos_sweeps = (
                                  self.velocities[:, 0] * -self.local_forward_directions[:, 0] +
                                  self.velocities[:, 1] * -self.local_forward_directions[:, 1] +
                                  self.velocities[:, 2] * -self.local_forward_directions[:, 2]
                          ) / self.velocity_magnitudes
        self.chord_perpendiculars = self.chords * self.cos_sweeps
        self.velocity_magnitude_perpendiculars = self.velocity_magnitudes * self.cos_sweeps
        self.Res_perpendicular = self.Res * self.cos_sweeps
        self.machs_perpendicular = self.machs * self.cos_sweeps

        CL_locals = [
            self.CL_functions[i](
                alpha=self.alpha_eff_perpendiculars[i],
                Re=self.Res_perpendicular[i],
                mach=self.machs_perpendicular[i],
            ) for i in range(self.n_panels)
        ]
        CDp_locals = [
            self.CD_functions[i](
                alpha=self.alpha_eff_perpendiculars[i],
                Re=self.Res_perpendicular[i],
                mach=self.machs_perpendicular[i],
            ) for i in range(self.n_panels)
        ]
        Cm_locals = [
            self.CM_functions[i](
                alpha=self.alpha_eff_perpendiculars[i],
                Re=self.Res_perpendicular[i],
                mach=self.machs_perpendicular[i],
            ) for i in range(self.n_panels)
        ]
        self.CL_locals = cas.vertcat(*CL_locals)
        self.CDp_locals = cas.vertcat(*CDp_locals)
        self.Cm_locals = cas.vertcat(*Cm_locals)

        self.Vi_cross_li = cas.horzcat(
            self.velocities[:, 1] * self.vortex_bound_leg[:, 2] - self.velocities[:, 2] * self.vortex_bound_leg[:, 1],
            self.velocities[:, 2] * self.vortex_bound_leg[:, 0] - self.velocities[:, 0] * self.vortex_bound_leg[:, 2],
            self.velocities[:, 0] * self.vortex_bound_leg[:, 1] - self.velocities[:, 1] * self.vortex_bound_leg[:, 0],
        )
        Vi_cross_li_magnitudes = np.sqrt(
            self.Vi_cross_li[:, 0] ** 2 +
            self.Vi_cross_li[:, 1] ** 2 +
            self.Vi_cross_li[:, 2] ** 2
        )

        # self.opti.subject_to([
        #     self.vortex_strengths * Vi_cross_li_magnitudes ==
        #     0.5 * self.velocity_magnitude_perpendiculars ** 2 * self.CL_locals * self.areas
        # ])
        self.opti.subject_to([
            self.vortex_strengths * Vi_cross_li_magnitudes * 2 / self.velocity_magnitude_perpendiculars ** 2 / self.areas ==
            self.CL_locals
        ])

    def _calculate_forces(self):

        if self.verbose:
            print("Calculating induced forces...")
        self.forces_inviscid_geometry = self.op_point.density * self.Vi_cross_li * self.vortex_strengths
        force_total_inviscid_geometry = cas.vertcat(
            cas.sum1(self.forces_inviscid_geometry[:, 0]),
            cas.sum1(self.forces_inviscid_geometry[:, 1]),
            cas.sum1(self.forces_inviscid_geometry[:, 2]),
        )  # Remember, this is in GEOMETRY AXES, not WIND AXES or BODY AXES.
        if self.run_symmetric:
            forces_inviscid_geometry_from_symmetry = cas.if_else(
                self.use_symmetry,
                reflect_over_XZ_plane(self.forces_inviscid_geometry),
                0
            )
            force_total_inviscid_geometry_from_symmetry = cas.vertcat(
                cas.sum1(forces_inviscid_geometry_from_symmetry[:, 0]),
                cas.sum1(forces_inviscid_geometry_from_symmetry[:, 1]),
                cas.sum1(forces_inviscid_geometry_from_symmetry[:, 2]),
            )
            force_total_inviscid_geometry += force_total_inviscid_geometry_from_symmetry
        self.force_total_inviscid_wind = cas.transpose(
            self.op_point.compute_rotation_matrix_wind_to_geometry()) @ force_total_inviscid_geometry

        if self.verbose:
            print("Calculating induced moments...")
        self.moments_inviscid_geometry = cas.cross(
            cas.transpose(cas.transpose(self.vortex_centers) - self.airplane.xyz_ref),
            self.forces_inviscid_geometry
        )
        moment_total_inviscid_geometry = cas.vertcat(
            cas.sum1(self.moments_inviscid_geometry[:, 0]),
            cas.sum1(self.moments_inviscid_geometry[:, 1]),
            cas.sum1(self.moments_inviscid_geometry[:, 2]),
        )  # Remember, this is in GEOMETRY AXES, not WIND AXES or BODY AXES.
        if self.run_symmetric:
            moments_inviscid_geometry_from_symmetry = cas.if_else(
                self.use_symmetry,
                -reflect_over_XZ_plane(self.moments_inviscid_geometry),
                0
            )
            moment_total_inviscid_geometry_from_symmetry = cas.vertcat(
                cas.sum1(moments_inviscid_geometry_from_symmetry[:, 0]),
                cas.sum1(moments_inviscid_geometry_from_symmetry[:, 1]),
                cas.sum1(moments_inviscid_geometry_from_symmetry[:, 2]),
            )
            moment_total_inviscid_geometry += moment_total_inviscid_geometry_from_symmetry
        self.moment_total_inviscid_wind = cas.transpose(
            self.op_point.compute_rotation_matrix_wind_to_geometry()) @ moment_total_inviscid_geometry

        if self.verbose:
            print("Calculating profile forces...")
        self.forces_profile_geometry = (
                (0.5 * self.op_point.density * self.velocity_magnitudes * self.velocities)
                * self.CDp_locals * self.areas
        )
        force_total_profile_geometry = cas.vertcat(
            cas.sum1(self.forces_profile_geometry[:, 0]),
            cas.sum1(self.forces_profile_geometry[:, 1]),
            cas.sum1(self.forces_profile_geometry[:, 2]),
        )
        if self.run_symmetric:
            forces_profile_geometry_from_symmetry = cas.if_else(
                self.use_symmetry,
                reflect_over_XZ_plane(self.forces_profile_geometry),
                0
            )
            force_total_profile_geometry_from_symmetry = cas.vertcat(
                cas.sum1(forces_profile_geometry_from_symmetry[:, 0]),
                cas.sum1(forces_profile_geometry_from_symmetry[:, 1]),
                cas.sum1(forces_profile_geometry_from_symmetry[:, 2]),
            )
            force_total_profile_geometry += force_total_profile_geometry_from_symmetry
        self.force_total_profile_wind = cas.transpose(
            self.op_point.compute_rotation_matrix_wind_to_geometry()) @ force_total_profile_geometry

        if self.verbose:
            print("Calculating profile moments...")
        self.moments_profile_geometry = cas.cross(
            cas.transpose(cas.transpose(self.vortex_centers) - self.airplane.xyz_ref),
            self.forces_profile_geometry
        )
        moment_total_profile_geometry = cas.vertcat(
            cas.sum1(self.moments_profile_geometry[:, 0]),
            cas.sum1(self.moments_profile_geometry[:, 1]),
            cas.sum1(self.moments_profile_geometry[:, 2]),
        )
        if self.run_symmetric:
            moments_profile_geometry_from_symmetry = cas.if_else(
                self.use_symmetry,
                -reflect_over_XZ_plane(self.moments_profile_geometry),
                0
            )
            moment_total_profile_geometry_from_symmetry = cas.vertcat(
                cas.sum1(moments_profile_geometry_from_symmetry[:, 0]),
                cas.sum1(moments_profile_geometry_from_symmetry[:, 1]),
                cas.sum1(moments_profile_geometry_from_symmetry[:, 2]),
            )
            moment_total_profile_geometry += moment_total_profile_geometry_from_symmetry
        self.moment_total_profile_wind = cas.transpose(
            self.op_point.compute_rotation_matrix_wind_to_geometry()) @ moment_total_profile_geometry

        if self.verbose:
            print("Calculating pitching moments...")
        bound_leg_YZ = self.vortex_bound_leg
        bound_leg_YZ[:, 0] = 0
        self.moments_pitching_geometry = (
                (0.5 * self.op_point.density * self.velocity_magnitudes ** 2) *
                self.Cm_locals * self.chords ** 2 * bound_leg_YZ
        )
        moment_total_pitching_geometry = cas.vertcat(
            cas.sum1(self.moments_pitching_geometry[:, 0]),
            cas.sum1(self.moments_pitching_geometry[:, 1]),
            cas.sum1(self.moments_pitching_geometry[:, 2]),
        )
        if self.run_symmetric:
            moments_pitching_geometry_from_symmetry = cas.if_else(
                self.use_symmetry,
                -reflect_over_XZ_plane(self.moments_pitching_geometry),
                0
            )
            moment_total_pitching_geometry_from_symmetry = cas.vertcat(
                cas.sum1(moments_pitching_geometry_from_symmetry[:, 0]),
                cas.sum1(moments_pitching_geometry_from_symmetry[:, 1]),
                cas.sum1(moments_pitching_geometry_from_symmetry[:, 2]),
            )
            moment_total_pitching_geometry += moment_total_pitching_geometry_from_symmetry
        self.moment_total_pitching_wind = cas.transpose(
            self.op_point.compute_rotation_matrix_wind_to_geometry()) @ moment_total_pitching_geometry

        if self.verbose:
            print("Calculating total forces and moments...")
        self.force_total_wind = self.force_total_inviscid_wind + self.force_total_profile_wind
        self.moment_total_wind = self.moment_total_inviscid_wind + self.moment_total_profile_wind

        # Calculate dimensional forces
        self.lift_force = -self.force_total_wind[2]
        self.drag_force = -self.force_total_wind[0]
        self.drag_force_induced = -self.force_total_inviscid_wind[0]
        self.drag_force_profile = -self.force_total_profile_wind[0]
        self.side_force = self.force_total_wind[1]

        # Calculate nondimensional forces
        q = self.op_point.dynamic_pressure()
        s_ref = self.airplane.s_ref
        b_ref = self.airplane.b_ref
        c_ref = self.airplane.c_ref
        self.CL = self.lift_force / q / s_ref
        self.CD = self.drag_force / q / s_ref
        self.CDi = self.drag_force_induced / q / s_ref
        self.CDp = self.drag_force_profile / q / s_ref
        self.CY = self.side_force / q / s_ref
        self.Cl = self.moment_total_wind[0] / q / s_ref / b_ref
        self.Cm = self.moment_total_wind[1] / q / s_ref / c_ref
        self.Cn = self.moment_total_wind[2] / q / s_ref / b_ref

        # Solves divide by zero error
        self.CL_over_CD = cas.if_else(self.CD == 0, 0, self.CL / self.CD)

    def calculate_Vij(self,
                      points,  # type: cas.MX
                      align_trailing_vortices_with_freestream=True,  # Otherwise, aligns with x-axis
                      ):
        # Calculates Vij, the velocity influence matrix (First index is collocation point number, second index is vortex number).
        # points: the list of points (Nx3) to calculate the velocity influence at.

        n_points = points.shape[0]

        # Make a and b vectors.
        # a: Vector from all collocation points to all horseshoe vortex left vertices.
        #   # First index is collocation point #, second is vortex #.
        # b: Vector from all collocation points to all horseshoe vortex right vertices.
        #   # First index is collocation point #, second is vortex #.
        a_x = points[:, 0] - cas.repmat(cas.transpose(self.left_vortex_vertices[:, 0]), n_points, 1)
        a_y = points[:, 1] - cas.repmat(cas.transpose(self.left_vortex_vertices[:, 1]), n_points, 1)
        a_z = points[:, 2] - cas.repmat(cas.transpose(self.left_vortex_vertices[:, 2]), n_points, 1)
        b_x = points[:, 0] - cas.repmat(cas.transpose(self.right_vortex_vertices[:, 0]), n_points, 1)
        b_y = points[:, 1] - cas.repmat(cas.transpose(self.right_vortex_vertices[:, 1]), n_points, 1)
        b_z = points[:, 2] - cas.repmat(cas.transpose(self.right_vortex_vertices[:, 2]), n_points, 1)

        if align_trailing_vortices_with_freestream:
            freestream_direction = self.op_point.compute_freestream_direction_geometry_axes()
            u_x = freestream_direction[0]
            u_y = freestream_direction[1]
            u_z = freestream_direction[2]
        else:
            u_x = 1
            u_y = 0
            u_z = 0

        # Do some useful arithmetic
        a_cross_b_x = a_y * b_z - a_z * b_y
        a_cross_b_y = a_z * b_x - a_x * b_z
        a_cross_b_z = a_x * b_y - a_y * b_x
        a_dot_b = a_x * b_x + a_y * b_y + a_z * b_z

        a_cross_u_x = a_y * u_z - a_z * u_y
        a_cross_u_y = a_z * u_x - a_x * u_z
        a_cross_u_z = a_x * u_y - a_y * u_x
        a_dot_u = a_x * u_x + a_y * u_y + a_z * u_z

        b_cross_u_x = b_y * u_z - b_z * u_y
        b_cross_u_y = b_z * u_x - b_x * u_z
        b_cross_u_z = b_x * u_y - b_y * u_x
        b_dot_u = b_x * u_x + b_y * u_y + b_z * u_z

        norm_a = np.sqrt(a_x ** 2 + a_y ** 2 + a_z ** 2)
        norm_b = np.sqrt(b_x ** 2 + b_y ** 2 + b_z ** 2)
        norm_a_inv = 1 / norm_a
        norm_b_inv = 1 / norm_b

        # Handle the special case where the collocation point is along a bound vortex leg
        a_cross_b_squared = (
                a_cross_b_x ** 2 +
                a_cross_b_y ** 2 +
                a_cross_b_z ** 2
        )
        a_dot_b = cas.if_else(a_cross_b_squared < 1e-8, a_dot_b + 1, a_dot_b)
        a_cross_u_squared = (
                a_cross_u_x ** 2 +
                a_cross_u_y ** 2 +
                a_cross_u_z ** 2
        )
        a_dot_u = cas.if_else(a_cross_u_squared < 1e-8, a_dot_u + 1, a_dot_u)
        b_cross_u_squared = (
                b_cross_u_x ** 2 +
                b_cross_u_y ** 2 +
                b_cross_u_z ** 2
        )
        b_dot_u = cas.if_else(b_cross_u_squared < 1e-8, b_dot_u + 1, b_dot_u)

        # Calculate Vij
        term1 = (norm_a_inv + norm_b_inv) / (norm_a * norm_b + a_dot_b)
        term2 = norm_a_inv / (norm_a - a_dot_u)
        term3 = norm_b_inv / (norm_b - b_dot_u)

        Vij_x = 1 / (4 * np.pi) * (
                a_cross_b_x * term1 +
                a_cross_u_x * term2 -
                b_cross_u_x * term3
        )
        Vij_y = 1 / (4 * np.pi) * (
                a_cross_b_y * term1 +
                a_cross_u_y * term2 -
                b_cross_u_y * term3
        )
        Vij_z = 1 / (4 * np.pi) * (
                a_cross_b_z * term1 +
                a_cross_u_z * term2 -
                b_cross_u_z * term3
        )
        if self.run_symmetric:  # If it's a symmetric problem, you've got to add the other side's influence.

            # If it is symmetric, re-do it with flipped coordinates

            # Make a and b vectors.
            # a: Vector from all collocation points to all horseshoe vortex left vertices.
            #   # First index is collocation point #, second is vortex #.
            # b: Vector from all collocation points to all horseshoe vortex right vertices.
            #   # First index is collocation point #, second is vortex #.
            a_x = points[:, 0] - cas.repmat(cas.transpose(self.right_vortex_vertices[:, 0]), n_points, 1)
            a_y = points[:, 1] - cas.repmat(cas.transpose(-self.right_vortex_vertices[:, 1]), n_points, 1)
            a_z = points[:, 2] - cas.repmat(cas.transpose(self.right_vortex_vertices[:, 2]), n_points, 1)
            b_x = points[:, 0] - cas.repmat(cas.transpose(self.left_vortex_vertices[:, 0]), n_points, 1)
            b_y = points[:, 1] - cas.repmat(cas.transpose(-self.left_vortex_vertices[:, 1]), n_points, 1)
            b_z = points[:, 2] - cas.repmat(cas.transpose(self.left_vortex_vertices[:, 2]), n_points, 1)

            # Do some useful arithmetic
            a_cross_b_x = a_y * b_z - a_z * b_y
            a_cross_b_y = a_z * b_x - a_x * b_z
            a_cross_b_z = a_x * b_y - a_y * b_x
            a_dot_b = a_x * b_x + a_y * b_y + a_z * b_z

            a_cross_u_x = a_y * u_z - a_z * u_y
            a_cross_u_y = a_z * u_x - a_x * u_z
            a_cross_u_z = a_x * u_y - a_y * u_x
            a_dot_u = a_x * u_x + a_y * u_y + a_z * u_z

            b_cross_u_x = b_y * u_z - b_z * u_y
            b_cross_u_y = b_z * u_x - b_x * u_z
            b_cross_u_z = b_x * u_y - b_y * u_x
            b_dot_u = b_x * u_x + b_y * u_y + b_z * u_z

            norm_a = np.sqrt(a_x ** 2 + a_y ** 2 + a_z ** 2)
            norm_b = np.sqrt(b_x ** 2 + b_y ** 2 + b_z ** 2)
            norm_a_inv = 1 / norm_a
            norm_b_inv = 1 / norm_b

            # Handle the special case where the collocation point is along a bound vortex leg
            a_cross_b_squared = (
                    a_cross_b_x ** 2 +
                    a_cross_b_y ** 2 +
                    a_cross_b_z ** 2
            )
            a_dot_b = cas.if_else(a_cross_b_squared < 1e-8, a_dot_b + 1, a_dot_b)
            a_cross_u_squared = (
                    a_cross_u_x ** 2 +
                    a_cross_u_y ** 2 +
                    a_cross_u_z ** 2
            )
            a_dot_u = cas.if_else(a_cross_u_squared < 1e-8, a_dot_u + 1, a_dot_u)
            b_cross_u_squared = (
                    b_cross_u_x ** 2 +
                    b_cross_u_y ** 2 +
                    b_cross_u_z ** 2
            )
            b_dot_u = cas.if_else(b_cross_u_squared < 1e-8, b_dot_u + 1, b_dot_u)

            # Calculate Vij
            term1 = (norm_a_inv + norm_b_inv) / (norm_a * norm_b + a_dot_b)
            term2 = norm_a_inv / (norm_a - a_dot_u)
            term3 = norm_b_inv / (norm_b - b_dot_u)

            Vij_x_from_symmetry = 1 / (4 * np.pi) * (
                    a_cross_b_x * term1 +
                    a_cross_u_x * term2 -
                    b_cross_u_x * term3
            )
            Vij_y_from_symmetry = 1 / (4 * np.pi) * (
                    a_cross_b_y * term1 +
                    a_cross_u_y * term2 -
                    b_cross_u_y * term3
            )
            Vij_z_from_symmetry = 1 / (4 * np.pi) * (
                    a_cross_b_z * term1 +
                    a_cross_u_z * term2 -
                    b_cross_u_z * term3
            )

            Vij_x += cas.transpose(cas.if_else(self.use_symmetry, cas.transpose(Vij_x_from_symmetry), 0))
            Vij_y += cas.transpose(cas.if_else(self.use_symmetry, cas.transpose(Vij_y_from_symmetry), 0))
            Vij_z += cas.transpose(cas.if_else(self.use_symmetry, cas.transpose(Vij_z_from_symmetry), 0))

        return Vij_x, Vij_y, Vij_z

    def calculate_fuselage_influences(self,
                                      points,  # type: cas.MX
                                      ):
        n_points = points.shape[0]

        fuselage_influences_x = cas.GenDM_zeros(n_points, 1)
        fuselage_influences_y = cas.GenDM_zeros(n_points, 1)
        fuselage_influences_z = cas.GenDM_zeros(n_points, 1)

        for fuse_num in range(len(self.airplane.fuselages)):
            this_fuse_centerline_points = self.fuse_centerline_points[fuse_num]
            this_fuse_radii = self.fuse_radii[fuse_num]

            dx = points[:, 0] - cas.repmat(cas.transpose(this_fuse_centerline_points[:, 0]), n_points, 1)
            dy = points[:, 1] - cas.repmat(cas.transpose(this_fuse_centerline_points[:, 1]), n_points, 1)
            dz = points[:, 2] - cas.repmat(cas.transpose(this_fuse_centerline_points[:, 2]), n_points, 1)

            # # Compressibility
            # dy *= self.beta
            # dz *= self.beta

            # For now, we're just putting a point source at the middle... # TODO make an actual line source
            source_x = (dx[:, 1:] + dx[:, :-1]) / 2
            source_y = (dy[:, 1:] + dy[:, :-1]) / 2
            source_z = (dz[:, 1:] + dz[:, :-1]) / 2

            areas = cas.pi * this_fuse_radii ** 2
            freestream_x_component = self.op_point.compute_freestream_velocity_geometry_axes()[
                0]  # TODO add in rotation corrections, add in doublets for alpha
            strengths = freestream_x_component * cas.diff(areas)

            denominator = 4 * cas.pi * (source_x ** 2 + source_y ** 2 + source_z ** 2) ** 1.5
            u = cas.transpose(strengths * cas.transpose(source_x / denominator))
            v = cas.transpose(strengths * cas.transpose(source_y / denominator))
            w = cas.transpose(strengths * cas.transpose(source_z / denominator))

            fuselage_influences_x += cas.sum2(u)
            fuselage_influences_y += cas.sum2(v)
            fuselage_influences_z += cas.sum2(w)

        fuselage_influences = cas.horzcat(
            fuselage_influences_x,
            fuselage_influences_y,
            fuselage_influences_z
        )

        return fuselage_influences

    def get_induced_velocity_at_point(self, point):
        if self.verbose and not self.opti.return_status() == 'Solve_Succeeded':
            print("WARNING: This method should only be used after a solution has been found!!!\n"
                  "Running anyway for debugging purposes - this is likely to not work.")

        Vij_x, Vij_y, Vij_z = self.calculate_Vij(point)

        # vortex_strengths = self.opti.debug.value(self.vortex_strengths)

        Vi_x = Vij_x @ self.vortex_strengths
        Vi_y = Vij_y @ self.vortex_strengths
        Vi_z = Vij_z @ self.vortex_strengths

        get = lambda x: self.opti.debug.value(x)
        Vi_x = get(Vi_x)
        Vi_y = get(Vi_y)
        Vi_z = get(Vi_z)

        Vi = np.vstack((Vi_x, Vi_y, Vi_z)).T

        return Vi

    def get_velocity_at_point(self, point):
        # Input: a Nx3 numpy array of points that you would like to know the velocities at.
        # Output: a Nx3 numpy array of the velocities at those points.

        Vi = self.get_induced_velocity_at_point(point) + self.calculate_fuselage_influences(
            point)  # TODO just a reminder, fuse added here

        freestream = self.op_point.compute_freestream_velocity_geometry_axes()

        V = cas.transpose(cas.transpose(Vi) + freestream)
        return V

    def calculate_streamlines(self,
                              seed_points=None,  # will be auto-calculated if not specified
                              n_steps=100,  # minimum of 2
                              length=None  # will be auto-calculated if not specified
                              ):

        if length is None:
            length = self.airplane.c_ref * 5
        if seed_points is None:
            seed_points = (self.back_left_vertices + self.back_right_vertices) / 2

        # Resolution
        length_per_step = length / n_steps

        # Initialize
        streamlines = [seed_points]

        # Iterate
        for step_num in range(1, n_steps):
            update_amount = self.get_velocity_at_point(streamlines[-1])
            norm_update_amount = np.sqrt(
                update_amount[:, 0] ** 2 + update_amount[:, 1] ** 2 + update_amount[:, 2] ** 2)
            update_amount = length_per_step * update_amount / norm_update_amount
            streamlines.append(streamlines[-1] + update_amount)

        self.streamlines = streamlines

    def draw(self,
             data_to_plot=None,
             data_name=None,
             show=True,
             draw_streamlines=True,
             recalculate_streamlines=False
             ):
        """
        Draws the solution. Note: Must be called on a SOLVED AeroProblem object.
        To solve an AeroProblem, use opti.solve(). To substitute a solved solution, use ap = ap.substitute_solution(sol).
        :return:
        """

        # TODO rewrite me

        if self.verbose:
            print("Drawing...")

        if self.verbose and not self.opti.return_status() == 'Solve_Succeeded':
            print("WARNING: This method should only be used after a solution has been found!\n"
                  "Running anyway for debugging purposes - this is likely to not work...")

        # Do substitutions
        get = lambda x: self.opti.debug.value(x)
        front_left_vertices = get(self.front_left_vertices)
        front_right_vertices = get(self.front_right_vertices)
        back_left_vertices = get(self.back_left_vertices)
        back_right_vertices = get(self.back_right_vertices)
        left_vortex_vertices = get(self.left_vortex_vertices)
        right_vortex_vertices = get(self.right_vortex_vertices)
        self.vortex_strengths = get(self.vortex_strengths)
        try:
            data_to_plot = get(data_to_plot)
        except NotImplementedError:
            pass

        if data_to_plot is None:
            CL_locals = get(self.CL_locals)
            chords = get(self.chords)
            c_ref = get(self.airplane.c_ref)
            data_name = "Cl * c / c_ref"
            data_to_plot = CL_locals * chords / c_ref

        fig = Figure3D()

        for index in range(len(front_left_vertices)):
            fig.add_quad(
                points=[
                    front_left_vertices[index, :],
                    front_right_vertices[index, :],
                    back_right_vertices[index, :],
                    back_left_vertices[index, :],
                ],
                intensity=data_to_plot[index],
                outline=True,
                mirror=self.run_symmetric and self.use_symmetry[index]
            )
            fig.add_line(
                points=[
                    left_vortex_vertices[index],
                    right_vortex_vertices[index]
                ],
                mirror=self.run_symmetric and self.use_symmetry[index]
            )

        # Fuselages
        for fuse_id in range(len(self.airplane.fuselages)):
            fuse = self.airplane.fuselages[fuse_id]  # type: Fuselage

            for xsec_id in range(len(fuse.xsecs) - 1):
                xsec_1 = fuse.xsecs[xsec_id]  # type: FuselageXSec
                xsec_2 = fuse.xsecs[xsec_id + 1]  # type: FuselageXSec

                r1 = xsec_1.radius
                r2 = xsec_2.radius
                points_1 = np.zeros((fuse.circumferential_panels, 3))
                points_2 = np.zeros((fuse.circumferential_panels, 3))
                for point_index in range(fuse.circumferential_panels):
                    rot = rotation_matrix_angle_axis(
                        2 * cas.pi * point_index / fuse.circumferential_panels,
                        [1, 0, 0],
                        True
                    ).toarray()
                    points_1[point_index, :] = rot @ np.array([0, 0, r1])
                    points_2[point_index, :] = rot @ np.array([0, 0, r2])
                points_1 = points_1 + np.array(xsec_1.xyz_c).reshape(-1)
                points_2 = points_2 + np.array(xsec_2.xyz_c).reshape(-1)

                for point_index in range(fuse.circumferential_panels):

                    fig.add_quad(points=[
                        points_1[(point_index) % fuse.circumferential_panels, :],
                        points_1[(point_index + 1) % fuse.circumferential_panels, :],
                        points_2[(point_index + 1) % fuse.circumferential_panels, :],
                        points_2[(point_index) % fuse.circumferential_panels, :],
                    ],
                        intensity=0,
                        mirror=fuse.symmetric,
                    )

        if draw_streamlines:
            if (not hasattr(self, 'streamlines')) or recalculate_streamlines:
                if self.verbose:
                    print("Calculating streamlines...")
                seed_points = (back_left_vertices + back_right_vertices) / 2
                self.calculate_streamlines(seed_points=seed_points)

            if self.verbose:
                print("Parsing streamline data...")
            n_streamlines = self.streamlines[0].shape[0]
            n_timesteps = len(self.streamlines)

            for streamlines_num in range(n_streamlines):
                streamline = [self.streamlines[ts][streamlines_num, :] for ts in range(n_timesteps)]
                fig.add_streamline(
                    points=streamline,
                    mirror=self.run_symmetric
                )

        return fig.draw(
            show=show,
            colorbar_title=data_name
        )
