import numpy as np
from aerosandbox import ExplicitAnalysis
from aerosandbox.geometry import *
from aerosandbox.aerodynamics.aero_3D.singularities.uniform_strength_horseshoe_singularities import \
    calculate_induced_velocity_horseshoe


class VortexLatticeMethod(ExplicitAnalysis):
    """
    An explicit (linear) vortex-lattice-method aerodynamics analysis.

    Usage example:
        >>>analysis = asb.VortexLatticeMethod(
        >>>    airplane=my_airplane,
        >>>    op_point=asb.OperatingPoint(
        >>>        velocity=100, # m/s
        >>>        alpha=5, # deg
        >>>        beta=4, # deg
        >>>        p=0.01, # rad/sec
        >>>        q=0.02, # rad/sec
        >>>        r=0.03, # rad/sec
        >>>    )
        >>>)
        >>>outputs = analysis.run()
    """

    def __init__(self,
                 airplane,  # type: Airplane
                 op_point,  # type: op_point
                 run_symmetric_if_possible=True,
                 verbose=False,
                 spanwise_resolution=10,
                 spanwise_spacing="cosine",
                 chordwise_resolution=10,
                 chordwise_spacing="cosine",
                 ):
        super().__init__()

        self.airplane = airplane
        self.op_point = op_point
        self.verbose = verbose
        self.spanwise_resolution = spanwise_resolution
        self.spanwise_spacing = spanwise_spacing
        self.chordwise_resolution = chordwise_resolution
        self.chordwise_spacing = chordwise_spacing

        ### Determine whether you should run the problem as symmetric
        self.run_symmetric = False
        if run_symmetric_if_possible:
            try:
                self.run_symmetric = (  # Satisfies assumptions
                        self.op_point.beta == 0 and
                        self.op_point.p == 0 and
                        self.op_point.r == 0 and
                        self.airplane.is_entirely_symmetric()
                )
            except RuntimeError:  # Required because beta, p, r, etc. may be non-numeric (e.g. opti variables)
                pass

    def run(self):
        if self.verbose:
            print("Meshing...")

        ##### Make Panels
        front_left_vertices = []
        front_right_vertices = []
        back_left_vertices = []
        back_right_vertices = []

        for wing in self.airplane.wings:
            points, faces = wing.mesh_thin_surface(
                method="quad",
                chordwise_resolution=self.chordwise_resolution,
                chordwise_spacing=self.chordwise_spacing,
                spanwise_resolution=self.spanwise_resolution,
                spanwise_spacing=self.spanwise_spacing,
                add_camber=True
            )
            front_left_vertices.append(points[faces[:, 0], :])
            front_right_vertices.append(points[faces[:, 1], :])
            back_left_vertices.append(points[faces[:, 3], :])
            back_right_vertices.append(points[faces[:, 2], :])

        front_left_vertices = np.concatenate(front_left_vertices)
        front_right_vertices = np.concatenate(front_right_vertices)
        back_left_vertices = np.concatenate(back_left_vertices)
        back_right_vertices = np.concatenate(back_right_vertices)

        ### Compute panel statistics
        diag1 = front_right_vertices - back_left_vertices
        diag2 = front_left_vertices - back_right_vertices
        cross = np.cross(diag1, diag2)
        cross_norm = (cross[:, 0] ** 2 + cross[:, 1] ** 2 + cross[:, 2] ** 2) ** 0.5
        normal_directions = cross / np.reshape(cross_norm, (-1, 1))
        areas = cross_norm / 2

        # Compute the location of points of interest on each panel
        left_vortex_vertices = 0.75 * front_left_vertices + 0.25 * back_left_vertices
        right_vortex_vertices = 0.75 * front_right_vertices + 0.25 * back_right_vertices
        vortex_centers = (left_vortex_vertices + right_vortex_vertices) / 2
        vortex_bound_leg = right_vortex_vertices - left_vortex_vertices
        collocation_points = 0.5 * (
                0.25 * front_left_vertices + 0.75 * back_left_vertices
        ) + 0.5 * (
                                     0.25 * front_right_vertices + 0.75 * back_right_vertices
                             )
        n_panels = collocation_points.shape[0]

        if self.verbose:
            print("Meshing complete!")

        ##### Setup Operating Point
        if self.verbose:
            print("Calculating the freestream influence...")
        steady_freestream_velocity = self.op_point.compute_freestream_velocity_geometry_axes()  # Direction the wind is GOING TO, in geometry axes coordinates
        steady_freestream_direction = steady_freestream_velocity / np.linalg.norm(steady_freestream_velocity)
        rotation_freestream_velocities = self.op_point.compute_rotation_velocity_geometry_axes(
            collocation_points)

        freestream_velocities = steady_freestream_velocity + rotation_freestream_velocities  # Nx3, represents the freestream velocity at each panel collocation point (c)

        freestream_influences = (
                freestream_velocities[:, 0] * normal_directions[:, 0] +
                freestream_velocities[:, 1] * normal_directions[:, 1] +
                freestream_velocities[:, 2] * normal_directions[:, 2]
        )

        ##### Setup Geometry
        ### Calculate AIC matrix
        if self.verbose:
            print("Calculating the collocation influence matrix...")

        def wide(array):
            return np.reshape(array, (1, -1))

        def tall(array):
            return np.reshape(array, (-1, 1))

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
            trailing_vortex_direction=steady_freestream_direction,
            gamma=1,
        )

        AIC = (
                u_collocations_unit * normal_directions[:, 0] +
                v_collocations_unit * normal_directions[:, 1] +
                w_collocations_unit * normal_directions[:, 2]
        )

        ##### Calculate Vortex Strengths
        if self.verbose:
            print("Calculating vortex strengths...")

        gamma = np.linalg.solve(AIC, -freestream_influences)

        ##### Calculate forces
        ### Calculate Near-Field Forces and Moments
        # Governing Equation: The force on a straight, small vortex filament is F = rho * V p l * gamma,
        # where rho is density, V is the velocity vector, p is the cross product operator,
        # l is the vector of the filament itself, and gamma is the circulation.

        if self.verbose:
            print("Calculating forces on each panel...")
        # Calculate the induced velocity at the center of each bound leg
        u_centers_induced, v_centers_induced, w_centers_induced = calculate_induced_velocity_horseshoe(
            x_field=tall(vortex_centers[:, 0]),
            y_field=tall(vortex_centers[:, 1]),
            z_field=tall(vortex_centers[:, 2]),
            x_left=wide(left_vortex_vertices[:, 0]),
            y_left=wide(left_vortex_vertices[:, 1]),
            z_left=wide(left_vortex_vertices[:, 2]),
            x_right=wide(right_vortex_vertices[:, 0]),
            y_right=wide(right_vortex_vertices[:, 1]),
            z_right=wide(right_vortex_vertices[:, 2]),
            trailing_vortex_direction=steady_freestream_direction,
            gamma=wide(gamma),
        )
        u_centers_induced = np.sum(u_centers_induced, axis=1)
        v_centers_induced = np.sum(v_centers_induced, axis=1)
        w_centers_induced = np.sum(w_centers_induced, axis=1)

        u_centers = u_centers_induced + freestream_velocities[:, 0]
        v_centers = v_centers_induced + freestream_velocities[:, 1]
        w_centers = w_centers_induced + freestream_velocities[:, 2]

        # Calculate forces_inviscid_geometry, the force on the ith panel. Note that this is in GEOMETRY AXES,
        # not WIND AXES or BODY AXES.
        # Vi_cross_li = np.cross(Vi, self.vortex_bound_leg, axis=1)
        Vi_cross_li = np.stack((
            v_centers * vortex_bound_leg[:, 2] - w_centers * vortex_bound_leg[:, 1],
            w_centers * vortex_bound_leg[:, 0] - u_centers * vortex_bound_leg[:, 2],
            u_centers * vortex_bound_leg[:, 1] - v_centers * vortex_bound_leg[:, 0],
        ), axis=1)
        forces_geometry = self.op_point.atmosphere.density() * Vi_cross_li * np.reshape(gamma, (-1, 1))

        # Calculate total forces and moments
        if self.verbose:
            print("Calculating total forces and moments...")
        force_geometry = np.sum(forces_geometry, axis=0)
        # Remember, this is in GEOMETRY AXES, not WIND AXES or BODY AXES.

        force_wind = np.transpose(
            self.op_point.compute_rotation_matrix_wind_to_geometry()) @ force_geometry
        # if self.verbose: print("Total aerodynamic forces (wind axes):", self.force_total_inviscid_wind)

        moments_geometry = np.cross(
            vortex_centers - np.reshape(self.airplane.xyz_ref, (1, -1)),
            forces_geometry
        )

        moment_geometry = np.sum(moments_geometry, axis=0)

        moment_wind = np.transpose(
            self.op_point.compute_rotation_matrix_wind_to_geometry()) @ moment_geometry

        # Calculate dimensional forces
        L = -force_wind[2]
        D = -force_wind[0]
        Y = force_wind[1]
        l = moment_wind[0]  # TODO review axes
        m = moment_wind[1]
        n = moment_wind[2]

        # Calculate nondimensional forces
        q = self.op_point.dynamic_pressure()
        s_ref = self.airplane.s_ref
        b_ref = self.airplane.b_ref
        c_ref = self.airplane.c_ref
        CL = L / q / s_ref
        CD = D / q / s_ref
        CY = Y / q / s_ref
        Cl = l / q / s_ref / b_ref
        Cm = m / q / s_ref / c_ref
        Cn = n / q / s_ref / b_ref

        return {
            "L" : L,
            "D" : D,
            "Y" : Y,
            "l" : l,
            "m" : m,
            "n" : n,
            "CL": CL,
            "CD": CD,
            "CY": CY,
            "Cl": Cl,
            "Cm": Cm,
            "Cn": Cn
        }

    def get_induced_velocity_at_point(self, point):
        if not self.opti.return_status() == 'Solve_Succeeded':
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

        Vi = self.get_induced_velocity_at_point(point)

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
            norm_update_amount = cas.sqrt(
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
            data_name = "Vortex Strengths"
            data_to_plot = get(self.vortex_strengths)

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
            )
            # fig.add_line( # Don't draw the quarter-chords
            #     points=[
            #         left_vortex_vertices[index],
            #         right_vortex_vertices[index]
            #     ],
            # )

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
                points_1 = points_1 + np.array(fuse.xyz_le).reshape(-1) + np.array(xsec_1.xyz_c).reshape(-1)
                points_2 = points_2 + np.array(fuse.xyz_le).reshape(-1) + np.array(xsec_2.xyz_c).reshape(-1)

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
                is_trailing_edge = np.array(self.is_trailing_edge, dtype=bool)
                seed_points = (back_left_vertices[is_trailing_edge] + back_right_vertices[is_trailing_edge]) / 2
                self.calculate_streamlines(seed_points=seed_points)

            if self.verbose:
                print("Parsing streamline data...")
            n_streamlines = self.streamlines[0].shape[0]
            n_timesteps = len(self.streamlines)

            for streamlines_num in range(n_streamlines):
                streamline = [self.streamlines[ts][streamlines_num, :] for ts in range(n_timesteps)]
                fig.add_streamline(
                    points=streamline,
                )

        return fig.draw(
            show=show,
            colorbar_title=data_name
        )


if __name__ == '__main__':
    ### Import Vanilla Airplane
    import aerosandbox as asb

    from pathlib import Path

    geometry_folder = Path(asb.__file__).parent.parent / "tutorial" / "04 - Geometry" / "example_geometry"

    import sys

    sys.path.insert(0, str(geometry_folder))

    from vanilla import airplane as vanilla

    ### Do the AVL run
    analysis = VortexLatticeMethod(
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

    res = analysis.run()

    for k, v in res.items():
        print(f"{str(k).rjust(10)} : {v:.4f}")
