from .aerodynamics import *


# noinspection PyAttributeOutsideInit
class vlm4(AeroProblem):
    # Ring vortex-lattice method aerodynamics code adapted from VLM3
    #
    # Notable improvements over VLM3:
    #   Ring vortices on all panels except the trailing edge panels, which have horseshoe vortices
    #       Ring vortices make the future implementation of unsteady methods feasible
    #
    # Usage:
    #   Set up a problem using the syntax in the AeroProblem constructor (e.g. "vlm4(airplane = a, op_point = op)")
    #   Call the run() method on the vlm4 object
    #   Access results in the command line, or through properties of the vlm2 class

    def run(self, verbose=True):
        # Runs a point analysis at the specified op-point.
        self.verbose = verbose

        if self.verbose:
            print("Running VLM4 calculation...")

        self.make_panels()
        self.setup_geometry()
        self.setup_operating_point()
        self.calculate_vortex_strengths()
        # self.calculate_forces()   # ToDo Uncomment this line after the calculate forces has been implemented

        if self.verbose:
            print("VLM4 calculation complete!")

    def make_panels(self):
        # Creates self.panel_coordinates_structured_list and self.wing_mcl_normals.

        if self.verbose:
            print("Meshing...")

        collocation_points = np.empty((0, 3))
        normal_directions = np.empty((0, 3))
        left_vortex_vertices = np.empty((0, 3))
        right_vortex_vertices = np.empty((0, 3))
        front_left_vertices = np.empty((0, 3))
        front_right_vertices = np.empty((0, 3))
        back_left_vertices = np.empty((0, 3))
        back_right_vertices = np.empty((0, 3))
        areas = np.empty(0)
        is_trailing_edge = np.empty(0, dtype=bool)
        xsec = None

        for wing_num in range(len(self.airplane.wings)):
            # For each wing, we want (where M is the number of chordwise panels, N is the number of spanwise panels):
            #   panel_coordinates_structured_list: M+1 x N+1 x 3; corners of every panel.
            #   normals_structured_list: M x N x 3; normal direction of each panel

            # Get the wing
            wing = self.airplane.wings[wing_num]

            # Define number of chordwise points
            n_chordwise_coordinates = wing.chordwise_panels + 1

            # Get the chordwise coordinates
            if wing.chordwise_spacing == 'uniform':
                nondim_chordwise_coordinates = np.linspace(0, 1, n_chordwise_coordinates)
            elif wing.chordwise_spacing == 'cosine':
                nondim_chordwise_coordinates = cosspace(0, 1, n_chordwise_coordinates)
            else:
                raise Exception("Bad value of wing.chordwise_spacing!")

            # Get corners of xsecs
            xsec_xyz_le = np.empty((0, 3))  # Nx3 array of leading edge points
            xsec_xyz_te = np.empty((0, 3))  # Nx3 array of trailing edge points
            for xsec in wing.xsecs:
                xsec_xyz_le = np.vstack((xsec_xyz_le, xsec.xyz_le + wing.xyz_le))
                xsec_xyz_te = np.vstack((xsec_xyz_te, xsec.xyz_te() + wing.xyz_le))

            # Get quarter-chord vector
            xsec_xyz_quarter_chords = 0.75 * xsec_xyz_le + 0.25 * xsec_xyz_te  # Nx3 array of quarter-chord points
            section_quarter_chords = (
                    xsec_xyz_quarter_chords[1:, :] -
                    xsec_xyz_quarter_chords[:-1, :]
            )  # Nx3 array of vectors connecting quarter-chords

            # Get directions for transforming 2D airfoil data to 3D

            # First, project quarter chords onto YZ plane and normalize.
            section_quarter_chords_proj = (section_quarter_chords[:, 1:] /
                                           np.expand_dims(np.linalg.norm(section_quarter_chords[:, 1:], axis=1),
                                                          axis=1))  # Nx2 array of quarter-chord vectors projected on YZ
            section_quarter_chords_proj = np.hstack(
                (np.zeros((section_quarter_chords_proj.shape[0], 1)),
                 section_quarter_chords_proj)
            )  # Convert back to a Nx3 array, since that's what we'll need later.
            # Then, construct the normal directions for each xsec.
            if len(wing.xsecs) > 2:  # Make normals for the inner xsecs, where we need to merge directions
                xsec_local_normal_inners = section_quarter_chords_proj[:-1, :] + section_quarter_chords_proj[1:, :]
                xsec_local_normal_inners = (xsec_local_normal_inners /
                                            np.expand_dims(np.linalg.norm(xsec_local_normal_inners, axis=1), axis=1)
                                            )
                xsec_local_normal = np.vstack((
                    section_quarter_chords_proj[0, :],
                    xsec_local_normal_inners,
                    section_quarter_chords_proj[-1, :]
                ))
            else:
                xsec_local_normal = np.vstack((
                    section_quarter_chords_proj[0, :],
                    section_quarter_chords_proj[-1, :]
                ))
            # xsec_local_normal is now a Nx3 array that represents the normal direction at each xsec.
            # Then, construct the back directions for each xsec.
            xsec_local_back = xsec_xyz_te - xsec_xyz_le  # aligned with chord
            xsec_chord = np.linalg.norm(xsec_local_back, axis=1)  # 1D vector, one per xsec
            xsec_local_back = (xsec_local_back /
                               np.expand_dims(xsec_chord, axis=1)
                               )
            # Then, construct the up direction for each xsec.
            xsec_local_up = np.cross(xsec_local_back, xsec_local_normal,
                                     axis=1)  # Nx3 array that represents the upwards direction at each xsec.

            # Get the scaling factor (airfoils at dihedral breaks need to be "taller" to compensate)
            xsec_scaling_factor = 1 / np.sqrt(
                (1 + np.sum(section_quarter_chords_proj[1:, :] * section_quarter_chords_proj[:-1, :], axis=1)
                 ) / 2
            )
            xsec_scaling_factor = np.hstack((1, xsec_scaling_factor, 1))  # TODO Make sure this is always right.

            # Make the panels for each section.

            for section_num in range(len(wing.xsecs) - 1):
                # Define the relevant cross sections
                inner_xsec = wing.xsecs[section_num]  # type is WingXSec
                outer_xsec = wing.xsecs[section_num + 1]  # type is WingXSec

                # Define the airfoils at each cross section
                inner_airfoil = inner_xsec.airfoil.add_control_surface(
                    deflection=inner_xsec.control_surface_deflection,
                    hinge_point=inner_xsec.control_surface_hinge_point
                )
                outer_airfoil = outer_xsec.airfoil.add_control_surface(
                    deflection=inner_xsec.control_surface_deflection,
                    # inner xsec dictates control surface deflections.
                    hinge_point=inner_xsec.control_surface_hinge_point
                )

                # Make the mean camber lines for each.
                inner_xsec_mcl_nondim = inner_airfoil.get_downsampled_mcl(nondim_chordwise_coordinates)
                outer_xsec_mcl_nondim = outer_airfoil.get_downsampled_mcl(nondim_chordwise_coordinates)
                # inner_xsec_mcl: First index is point number, second index is xyz.
                inner_xsec_mcl = xsec_xyz_le[section_num, :] + (xsec_local_back[section_num, :] *
                                                                np.expand_dims(inner_xsec_mcl_nondim[:, 0], 1) *
                                                                xsec_chord[section_num] + xsec_local_up[section_num, :]
                                                                * np.expand_dims(inner_xsec_mcl_nondim[:, 1], 1) *
                                                                xsec_chord[section_num] *
                                                                xsec_scaling_factor[section_num]
                                                                )
                outer_xsec_mcl = xsec_xyz_le[section_num + 1, :] + (
                        xsec_local_back[section_num + 1, :] * np.expand_dims(outer_xsec_mcl_nondim[:, 0], 1) *
                        xsec_chord[section_num + 1] +
                        xsec_local_up[section_num + 1, :] * np.expand_dims(outer_xsec_mcl_nondim[:, 1], 1) * xsec_chord[
                            section_num + 1] * xsec_scaling_factor[
                            section_num + 1]
                )

                # Define number of spanwise points
                n_spanwise_coordinates = xsec.spanwise_panels + 1

                # Get the spanwise coordinates
                if xsec.spanwise_spacing == 'uniform':
                    nondim_spanwise_coordinates = np.linspace(0, 1, n_spanwise_coordinates)
                elif xsec.spanwise_spacing == 'cosine':
                    nondim_spanwise_coordinates = cosspace(n_points=n_spanwise_coordinates)
                else:
                    raise Exception("Bad value of section.spanwise_spacing!")

                # Make section_mcl_coordinates: MxNx3 array of mean camberline coordinates.
                # First index is chordwise location, second index is spanwise location, third is xyz.
                section_mcl_coordinates = (
                        np.expand_dims(np.expand_dims((1 - nondim_spanwise_coordinates), 0),
                                       2) * np.expand_dims(
                    inner_xsec_mcl, 1) +
                        np.expand_dims(np.expand_dims(nondim_spanwise_coordinates, 0), 2) * np.expand_dims(
                    outer_xsec_mcl, 1)
                )  # TODO this is not strictly speaking correct, only true in the limit of small twist angles.

                # Compute corners of each panel
                front_inner_coordinates = section_mcl_coordinates[:-1, :-1, :]
                front_outer_coordinates = section_mcl_coordinates[:-1, 1:, :]
                back_inner_coordinates = section_mcl_coordinates[1:, :-1, :]
                back_outer_coordinates = section_mcl_coordinates[1:, 1:, :]
                section_is_trailing_edge = np.vstack((
                    np.zeros((wing.chordwise_panels - 1, xsec.spanwise_panels), dtype=bool),
                    np.ones((1, xsec.spanwise_panels), dtype=bool)
                ))

                # Reshape
                front_inner_coordinates = np.reshape(front_inner_coordinates, (-1, 3), order='F')
                front_outer_coordinates = np.reshape(front_outer_coordinates, (-1, 3), order='F')
                back_inner_coordinates = np.reshape(back_inner_coordinates, (-1, 3), order='F')
                back_outer_coordinates = np.reshape(back_outer_coordinates, (-1, 3), order='F')
                section_is_trailing_edge = np.reshape(section_is_trailing_edge, (-1), order='F')

                # Calculate panel normals and areas via diagonals
                diag1 = front_outer_coordinates - back_inner_coordinates
                diag2 = front_inner_coordinates - back_outer_coordinates
                diag_cross = np.cross(diag1, diag2, axis=1)
                diag_cross_norm = np.linalg.norm(diag_cross, axis=1)
                normals_to_add = diag_cross / np.expand_dims(diag_cross_norm, axis=1)
                areas_to_add = diag_cross_norm / 2

                # Make the panel data
                collocations_to_add = (
                        0.5 * (0.25 * front_inner_coordinates + 0.75 * back_inner_coordinates) +
                        0.5 * (0.25 * front_outer_coordinates + 0.75 * back_outer_coordinates)
                )
                inner_vortex_vertices_to_add = 0.75 * front_inner_coordinates + 0.25 * back_inner_coordinates
                outer_vortex_vertices_to_add = 0.75 * front_outer_coordinates + 0.25 * back_outer_coordinates

                # Append to the lists of panel data (c, n, lv, rv, etc.)
                front_left_vertices = np.vstack((
                    front_left_vertices,
                    front_inner_coordinates
                ))
                front_right_vertices = np.vstack((
                    front_right_vertices,
                    front_outer_coordinates
                ))
                back_left_vertices = np.vstack((
                    back_left_vertices,
                    back_inner_coordinates
                ))
                back_right_vertices = np.vstack((
                    back_right_vertices,
                    back_outer_coordinates
                ))
                areas = np.hstack((
                    areas,
                    areas_to_add
                ))
                is_trailing_edge = np.hstack((
                    is_trailing_edge,
                    section_is_trailing_edge
                ))
                collocation_points = np.vstack((
                    collocation_points,
                    collocations_to_add
                ))
                normal_directions = np.vstack((
                    normal_directions,
                    normals_to_add
                ))
                left_vortex_vertices = np.vstack((
                    left_vortex_vertices,
                    inner_vortex_vertices_to_add
                ))
                right_vortex_vertices = np.vstack((
                    right_vortex_vertices,
                    outer_vortex_vertices_to_add
                ))

                # Handle symmetry
                if wing.symmetric:
                    if inner_xsec.control_surface_type == "asymmetric":
                        # Define the airfoils at each cross section
                        inner_airfoil = inner_xsec.airfoil.add_control_surface(
                            deflection=-inner_xsec.control_surface_deflection,
                            hinge_point=inner_xsec.control_surface_hinge_point
                        )
                        outer_airfoil = outer_xsec.airfoil.add_control_surface(
                            deflection=-inner_xsec.control_surface_deflection,
                            # inner xsec dictates control surface deflections.
                            hinge_point=inner_xsec.control_surface_hinge_point
                        )

                        # Make the mean camber lines for each.
                        inner_xsec_mcl_nondim = inner_airfoil.get_downsampled_mcl(nondim_chordwise_coordinates)
                        outer_xsec_mcl_nondim = outer_airfoil.get_downsampled_mcl(nondim_chordwise_coordinates)
                        # inner_xsec_mcl: First index is point number, second index is xyz.
                        inner_xsec_mcl = xsec_xyz_le[section_num, :] + (
                                xsec_local_back[section_num, :] * np.expand_dims(inner_xsec_mcl_nondim[:, 0], 1) *
                                xsec_chord[
                                    section_num] +
                                xsec_local_up[section_num, :] * np.expand_dims(inner_xsec_mcl_nondim[:, 1], 1) *
                                xsec_chord[
                                    section_num] * xsec_scaling_factor[section_num]
                        )
                        outer_xsec_mcl = xsec_xyz_le[section_num + 1, :] + (
                                xsec_local_back[section_num + 1, :] * np.expand_dims(outer_xsec_mcl_nondim[:, 0], 1) *
                                xsec_chord[section_num + 1] +
                                xsec_local_up[section_num + 1, :] * np.expand_dims(outer_xsec_mcl_nondim[:, 1], 1) *
                                xsec_chord[
                                    section_num + 1] * xsec_scaling_factor[
                                    section_num + 1]
                        )

                        # Make section_mcl_coordinates: MxNx3 array of mean camberline coordinates.
                        # First index is chordwise location, second index is spanwise location, third is xyz.
                        section_mcl_coordinates = (
                                np.expand_dims(np.expand_dims((1 - nondim_spanwise_coordinates), 0),
                                               2) * np.expand_dims(
                            inner_xsec_mcl, 1) +
                                np.expand_dims(np.expand_dims(nondim_spanwise_coordinates, 0), 2) * np.expand_dims(
                            outer_xsec_mcl, 1)
                        )  # TODO this is not strictly speaking correct, only true in the limit of small twist angles.

                        # Compute corners of each panel
                        front_inner_coordinates = section_mcl_coordinates[:-1, :-1, :]
                        front_outer_coordinates = section_mcl_coordinates[:-1, 1:, :]
                        back_inner_coordinates = section_mcl_coordinates[1:, :-1, :]
                        back_outer_coordinates = section_mcl_coordinates[1:, 1:, :]

                        # Reshape
                        front_inner_coordinates = np.reshape(front_inner_coordinates, (-1, 3), order='F')
                        front_outer_coordinates = np.reshape(front_outer_coordinates, (-1, 3), order='F')
                        back_inner_coordinates = np.reshape(back_inner_coordinates, (-1, 3), order='F')
                        back_outer_coordinates = np.reshape(back_outer_coordinates, (-1, 3), order='F')

                        # Calculate panel normals and areas via diagonals
                        diag1 = front_outer_coordinates - back_inner_coordinates
                        diag2 = front_inner_coordinates - back_outer_coordinates
                        diag_cross = np.cross(diag1, diag2, axis=1)
                        diag_cross_norm = np.linalg.norm(diag_cross, axis=1)
                        normals_to_add = diag_cross / np.expand_dims(diag_cross_norm, axis=1)
                        areas_to_add = diag_cross_norm / 2

                        # Make the panels and append them to the lists of panel data (c, n, lv, rv, etc.)
                        collocations_to_add = (
                                0.5 * (0.25 * front_inner_coordinates + 0.75 * back_inner_coordinates) +
                                0.5 * (0.25 * front_outer_coordinates + 0.75 * back_outer_coordinates)
                        )
                        inner_vortex_vertices_to_add = 0.75 * front_inner_coordinates + 0.25 * back_inner_coordinates
                        outer_vortex_vertices_to_add = 0.75 * front_outer_coordinates + 0.25 * back_outer_coordinates

                    front_left_vertices = np.vstack((
                        front_left_vertices,
                        reflect_over_XZ_plane(front_outer_coordinates)
                    ))
                    front_right_vertices = np.vstack((
                        front_right_vertices,
                        reflect_over_XZ_plane(front_inner_coordinates)
                    ))
                    back_left_vertices = np.vstack((
                        back_left_vertices,
                        reflect_over_XZ_plane(back_outer_coordinates)
                    ))
                    back_right_vertices = np.vstack((
                        back_right_vertices,
                        reflect_over_XZ_plane(back_inner_coordinates)
                    ))
                    areas = np.hstack((
                        areas,
                        areas_to_add
                    ))
                    is_trailing_edge = np.hstack((
                        is_trailing_edge,
                        section_is_trailing_edge
                    ))
                    collocation_points = np.vstack((
                        collocation_points,
                        reflect_over_XZ_plane(collocations_to_add)
                    ))
                    normal_directions = np.vstack((
                        normal_directions,
                        reflect_over_XZ_plane(normals_to_add)
                    ))
                    left_vortex_vertices = np.vstack((
                        left_vortex_vertices,
                        reflect_over_XZ_plane(outer_vortex_vertices_to_add)
                    ))
                    right_vortex_vertices = np.vstack((
                        right_vortex_vertices,
                        reflect_over_XZ_plane(inner_vortex_vertices_to_add)
                    ))

        # Write to self object
        self.front_left_vertices = front_left_vertices
        self.front_right_vertices = front_right_vertices
        self.back_left_vertices = back_left_vertices
        self.back_right_vertices = back_right_vertices
        self.areas = areas
        self.is_trailing_edge = is_trailing_edge
        self.collocation_points = collocation_points
        self.normal_directions = normal_directions
        # self.left_vortex_vertices = left_vortex_vertices      # ToDo Delete this line if it is unnecessary
        # self.right_vortex_vertices = right_vortex_vertices    # ToDo Delete this line if it is unnecessary

        # Find horseshoe vortices
        self.left_horseshoe_vortex_vertices = back_left_vertices[is_trailing_edge]
        self.right_horseshoe_vortex_vertices = back_right_vertices[is_trailing_edge]

        # Do final processing for later use
        # ToDo Delete the following line if it is unnecessary
        # self.vortex_centers = (self.left_vortex_vertices + self.right_vortex_vertices) / 2
        # The following line calculates the center of the bound leg of each horseshoe vortex
        self.horseshoe_vortex_centers = (self.left_horseshoe_vortex_vertices + self.right_horseshoe_vortex_vertices) / 2
        # ToDo Delete the following line if it is unnecessary
        # self.vortex_bound_leg = (self.right_vortex_vertices - self.left_vortex_vertices)
        self.n_panels = len(self.collocation_points)
        self.n_horseshoes = len(self.left_horseshoe_vortex_vertices)  # Calculates the number of horseshoe vortices

        if self.verbose:
            print("Meshing complete!")
        # -----------------------------------------------------
        # Review of the important things that have been done up to this point:
        #   We made panel_coordinates_structured_list, a MxNx3 array describing a structured quadrilateral mesh of the
        #   wing's mean camber surface.
        #       For reference: first index is chordwise coordinate, second index is spanwise coordinate, and third index
        #       is xyz.
        #   We made normals_structured_list, a MxNx3 array describing the normal direction of the mean camber surface at
        #   the collocation point.
        #       For reference: first index is chordwise coordinate, second index is spanwise coordinate, and third index
        #       is xyz.
        #       Takes into account control surface deflections
        #   Both panel_coordinates_structured_list and normals_structured_list have been appended to lists of ndarrays
        #   within the vlm2 class, accessible at self.panel_coordinates_structured_list and
        #   self.normals_structured_list, respectively.
        #   Control surface handling:
        #       Control surfaces are implemented into normal directions as intended.
        #   Symmetry handling:
        #       All symmetric wings have been split into separate halves.
        #       All wing halves have their spanwise coordinates labeled from the left side of the airplane to the right.
        #       Control surface deflection symmetry has been handled; this is encoded into the normal directions.
        #   And best of all, it's all verified to be reverse-mode AD compatible!!!

    def setup_geometry(self):
        # Calculate AIC matrix
        if self.verbose:
            print("Calculating the collocation influence matrix...")

        self.Vij_collocations = self.calculate_Vij(self.collocation_points)
        # Vij_collocations: [points, vortices, xyz]
        # n: [points, xyz]

        # AIC = (Vij * normal vectors)
        self.AIC = np.sum(
            self.Vij_collocations * np.expand_dims(self.normal_directions, 1),
            axis=2
        )

        if self.verbose:
            print("Collocation influence matrix calculated!")

        # ToDo Uncomment and modify after calculate_forces and calculate_delta_cp functions are complete
        """# Calculate Vij at vortex centers for force calculation
        if self.verbose:
            print("Calculating the vortex center influence matrix...")

        # self.Vij_centers = self.calculate_Vij(self.vortex_centers)

        if self.verbose:
            print("Vortex center influence matrix calculated!")"""

        # # LU Decomposition on AIC
        # -------------------------
        # Unfortunately, I don't think we can use sp_linalg.lu_factor with autograd, so we'll have to do a direct solve
        # for every op-point instead of saving an LU-factorization and reusing it.
        # This isn't the worst, though, since the solution time is very small compared to the AIC calculation time, and
        # autograd gives us good gradients to use for op-point trimming and other things. So that's nice. I guess.
        # The long-term solution here would be to write a vector-jacobian product in autograd for lu_factor and lu_
        # solve.

    def setup_operating_point(self):  # TODO Check this method.

        if self.verbose:
            print("Calculating the freestream influence...")
        self.steady_freestream_velocity = np.expand_dims(self.op_point.compute_freestream_velocity_geometry_axes(),
                                                         0)  # Direction the wind is GOING TO, in geometry axes
        # coordinates
        self.rotation_freestream_velocities = self.op_point.compute_rotation_velocity_geometry_axes(
            self.collocation_points)

        self.freestream_velocities = self.steady_freestream_velocity + self.rotation_freestream_velocities  # Nx3,
        # represents the freestream velocity at each panel collocation point (c)

        self.freestream_influences = np.sum(self.freestream_velocities * self.normal_directions, axis=1)

    def calculate_vortex_strengths(self):
        # Calculate Vortex Strengths

        # Governing Equation: AIC @ Gamma + freestream_influence = 0
        if self.verbose:
            print("Calculating vortex strengths...")

        self.vortex_strengths = np.linalg.solve(self.AIC, -self.freestream_influences)

        self.ring_vortex_strengths = self.vortex_strengths[:self.n_panels]
        self.horseshoe_vortex_strengths = self.vortex_strengths[self.n_panels:]

        if self.verbose:
            print("Calculating vortex strengths completed!")
            print("Net horseshoe vortex strength: ", np.sum(self.horseshoe_vortex_strengths))

    # ToDo Rewrite this function to work with ring vortices
    """def calculate_forces(self):
        # Calculate Near-Field Forces and Moments

        # Governing Equation: The force on a straight, small vortex filament is F = rho * V x l * gamma,
        # where rho is density, V is the velocity vector, x is the cross product operator,
        # l is the vector of the filament itself, and gamma is the circulation.

        if self.verbose:
            print("Calculating forces on each panel...")

        # Calculate Vi (local velocity at the ith vortex center point)
        Vi_x = self.Vij_centers[:, :, 0] @ self.vortex_strengths + self.freestream_velocities[:, 0]
        Vi_y = self.Vij_centers[:, :, 1] @ self.vortex_strengths + self.freestream_velocities[:, 1]
        Vi_z = self.Vij_centers[:, :, 2] @ self.vortex_strengths + self.freestream_velocities[:, 2]
        Vi_x = np.expand_dims(Vi_x, axis=1)
        Vi_y = np.expand_dims(Vi_y, axis=1)
        Vi_z = np.expand_dims(Vi_z, axis=1)
        Vi = np.hstack((Vi_x, Vi_y, Vi_z))

        # Calculate Fi_geometry, the force on the ith panel. Note that this is in GEOMETRY AXES,
        # not WIND AXES or BODY AXES.
        density = self.op_point.density
        Vi_cross_li = np.cross(Vi, self.vortex_bound_leg, axis=1)
        vortex_strengths_expanded = np.expand_dims(self.vortex_strengths, axis=1)
        self.Fi_geometry = density * Vi_cross_li * vortex_strengths_expanded

        if self.verbose:
            print("Finished calculating forces on each panel!")

        # Calculate total forces and moments
        if self.verbose:
            print("Calculating total forces and moments...")

        self.Ftotal_geometry = np.sum(self.Fi_geometry,
                                      axis=0)  # Remember, this is in geometry axes, not wind axes or body axes

        self.Ftotal_wind = np.transpose(self.op_point.compute_rotation_matrix_wind_to_geometry()) @ self.Ftotal_geometry

        self.Mtotal_geometry = np.sum(np.cross(self.vortex_centers - self.airplane.xyz_ref, self.Fi_geometry),
                                      axis=0)
        self.Mtotal_wind = np.transpose(self.op_point.compute_rotation_matrix_wind_to_geometry()) @ self.Mtotal_geometry

        # Calculate nondimensional forces
        q = self.op_point.dynamic_pressure()
        s_ref = self.airplane.s_ref
        b_ref = self.airplane.b_ref
        c_ref = self.airplane.c_ref
        self.CL = -self.Ftotal_wind[2] / q / s_ref
        self.CDi = -self.Ftotal_wind[0] / q / s_ref
        self.CY = self.Ftotal_wind[1] / q / s_ref
        self.Cl = self.Mtotal_wind[0] / q / b_ref
        self.Cm = self.Mtotal_wind[1] / q / c_ref
        self.Cn = self.Mtotal_wind[2] / q / b_ref

        # Calculate nondimensional moments

        # Solves divide by zero error
        if self.CDi == 0:
            self.CL_over_CDi = 0
        else:
            self.CL_over_CDi = self.CL / self.CDi

        if self.verbose:
            print("\nForces\n-----")
            print("CL: ", self.CL)
            print("CDi: ", self.CDi)
            print("CY: ", self.CY)
            print("CL/CDi: ", self.CL_over_CDi)
            print("\nMoments\n-----")
            print("Cl: ", self.Cl)
            print("Cm: ", self.Cm)
            print("Cn: ", self.Cn)
            print("Finished calculating total forces and moments!")"""

    def calculate_Vij(self, points):

        # Calculates Vij, the velocity influence matrix (First index is collocation point number, second index is vortex
        # number).
        # points: the list of points (Nx3) to calculate the velocity influence at.

        Vij_ring_vortices = self.calculate_Vij_ring_vortices(
            points)  # Calculates the section of Vij corresponding to the doublets (ring vortices)
        Vij_horseshoe_vortices = self.calculate_Vij_horseshoe_vortices(
            points)  # Calculates the section of Vij corresponding to the trailing horseshoe vortices
        # Vij = np.hstack((Vij_ring_vortices, Vij_horseshoe_vortices))

        Vij = np.zeros((self.n_panels, self.n_panels, 3))
        np.place(
            Vij,
            np.tile(np.expand_dims(np.expand_dims(np.logical_not(self.is_trailing_edge), 0), 2),(self.n_panels, 1, 3)),
            Vij_ring_vortices
        )
        np.place(
            Vij,
            np.tile(np.expand_dims(np.expand_dims((self.is_trailing_edge), 0), 2),(self.n_panels, 1, 3)),
            Vij_horseshoe_vortices
        )

        # ToDo Delete the following code if it is unnecessary
        """"# Make lv and rv
        left_vortex_vertices = self.left_vortex_vertices
        right_vortex_vertices = self.right_vortex_vertices

        points = np.reshape(points, (-1, 3))
        n_points = len(points)
        n_vortices = self.n_panels

        # Make a and b vectors.
        # a: Vector from all collocation points to all horseshoe vortex left  vertices, NxNx3.
        #       First index is collocation point #, second is vortex #, and third is xyz. N=n_panels
        # b: Vector from all collocation points to all horseshoe vortex right vertices, NxNx3.
        #       First index is collocation point #, second is vortex #, and third is xyz. N=n_panels
        # a[i,j,:] = c[i,:] - lv[j,:]
        # b[i,j,:] = c[i,:] - rv[j,:]
        points = np.expand_dims(points, 1)
        a = points - left_vortex_vertices
        b = points - right_vortex_vertices
        # x_hat = np.zeros([n_points, n_vortices, 3])
        # x_hat[:, :, 0] = 1

        # Do some useful arithmetic
        a_cross_b = np.cross(a, b, axis=2)
        a_dot_b = np.einsum('ijk,ijk->ij', a, b)

        a_cross_x = np.stack((
            np.zeros((n_points, n_vortices)),
            a[:, :, 2],
            -a[:, :, 1]
        ), axis=2)
        a_dot_x = a[:, :, 0]

        b_cross_x = np.stack((
            np.zeros((n_points, n_vortices)),
            b[:, :, 2],
            -b[:, :, 1]
        ), axis=2)
        b_dot_x = b[:, :, 0]  # np.sum(b * x_hat,axis=2)

        norm_a = np.linalg.norm(a, axis=2)
        norm_b = np.linalg.norm(b, axis=2)
        norm_a_inv = 1 / norm_a
        norm_b_inv = 1 / norm_b

        # Check for the special case where the collocation point is along the bound vortex leg
        # Find where cross product is near zero, and set the dot product to infinity so that the value of the bound term
        # is zero.
        bound_vortex_singularity_indices = (
                np.einsum('ijk,ijk->ij', a_cross_b, a_cross_b)  # norm(a_cross_b) ** 2
                < 3.0e-16)
        a_dot_b = a_dot_b + bound_vortex_singularity_indices
        left_vortex_singularity_indices = (
                np.einsum('ijk,ijk->ij', a_cross_x, a_cross_x)
                < 3.0e-16
        )
        a_dot_x = a_dot_x + left_vortex_singularity_indices
        right_vortex_singularity_indices = (
                np.einsum('ijk,ijk->ij', b_cross_x, b_cross_x)
                < 3.0e-16
        )
        b_dot_x = b_dot_x + right_vortex_singularity_indices

        # Calculate Vij
        term1 = (norm_a_inv + norm_b_inv) / (norm_a * norm_b + a_dot_b)
        term2 = norm_a_inv / (norm_a - a_dot_x)
        term3 = norm_b_inv / (norm_b - b_dot_x)
        term1 = np.expand_dims(term1, 2)
        term2 = np.expand_dims(term2, 2)
        term3 = np.expand_dims(term3, 2)

        Vij = 1 / (4 * np.pi) * (
                a_cross_b * term1 +
                a_cross_x * term2 -
                b_cross_x * term3
        )

        return Vij"""

        return Vij

    def calculate_Vij_ring_vortices(self, points):
        # Calculates the doublet part of Vij, the velocity influence matrix (First index is collocation point number,
        # second index is vortex number).
        # points: the list of points (Nx3) to calculate the velocity influence at.

        # Data cleanup
        points = np.reshape(points, (-1, 3))

        # Make v1, v2, v3, and v4 vectors.
        # Each vector goes from all collocation points to one type of vertex (front left, front right, etc.). NxNx3.
        #   # First index is collocation point #, second is vortex #, and third is xyz. N=n_panels
        # v1: corresponds to front left vertices
        # v2: corresponds to front right vertices
        # v3: corresponds to back right vertices
        # v4: corresponds to back left vertices
        # Example: v1[i,j,:] = collocation_points[i,:] - front_left_vertices[j,:]

        points = np.expand_dims(points, 1)
        v1 = points - self.front_left_vertices[np.logical_not(self.is_trailing_edge)]
        v2 = points - self.front_right_vertices[np.logical_not(self.is_trailing_edge)]
        v3 = points - self.back_right_vertices[np.logical_not(self.is_trailing_edge)]
        v4 = points - self.back_left_vertices[np.logical_not(self.is_trailing_edge)]

        # x_hat = np.zeros([n_points, n_vortices, 3])
        # x_hat[:, :, 0] = 1

        # Do some useful arithmetic
        v1_cross_v2 = np.cross(v1, v2, axis=2)
        v2_cross_v3 = np.cross(v2, v3, axis=2)
        v3_cross_v4 = np.cross(v3, v4, axis=2)
        v4_cross_v1 = np.cross(v4, v1, axis=2)
        v1_dot_v2 = np.einsum('ijk,ijk->ij', v1, v2)
        v2_dot_v3 = np.einsum('ijk,ijk->ij', v2, v3)
        v3_dot_v4 = np.einsum('ijk,ijk->ij', v3, v4)
        v4_dot_v1 = np.einsum('ijk,ijk->ij', v4, v1)
        norm_v1 = np.linalg.norm(v1, axis=2)
        norm_v2 = np.linalg.norm(v2, axis=2)
        norm_v3 = np.linalg.norm(v3, axis=2)
        norm_v4 = np.linalg.norm(v4, axis=2)
        norm_v1_inv = 1 / norm_v1
        norm_v2_inv = 1 / norm_v2
        norm_v3_inv = 1 / norm_v3
        norm_v4_inv = 1 / norm_v4

        # Check for the special case where the collocation point is along the bound vortex leg
        # Find where cross product is near zero, and set the dot product to infinity so that the value of the bound term
        # is zero.
        v1_v2_singularity_indices = (
                np.einsum('ijk,ijk->ij', v1_cross_v2, v1_cross_v2)  # norm(cross_product)^2
                < 3.0e-16)
        v1_dot_v2 = v1_dot_v2 + v1_v2_singularity_indices

        v2_v3_singularity_indices = (
                np.einsum('ijk,ijk->ij', v2_cross_v3, v2_cross_v3)  # norm(cross_product)^2
                < 3.0e-16)
        v2_dot_v3 = v2_dot_v3 + v2_v3_singularity_indices

        v3_v4_singularity_indices = (
                np.einsum('ijk,ijk->ij', v3_cross_v4, v3_cross_v4)  # norm(cross_product)^2
                < 3.0e-16)
        v3_dot_v4 = v3_dot_v4 + v3_v4_singularity_indices

        v4_v1_singularity_indices = (
                np.einsum('ijk,ijk->ij', v4_cross_v1, v4_cross_v1)  # norm(cross_product)^2
                < 3.0e-16)
        v4_dot_v1 = v4_dot_v1 + v4_v1_singularity_indices

        # Calculate Vij
        term1 = (norm_v1_inv + norm_v2_inv) / (norm_v1 * norm_v2 + v1_dot_v2)
        term1 = np.expand_dims(term1, 2)
        term2 = (norm_v2_inv + norm_v3_inv) / (norm_v2 * norm_v3 + v2_dot_v3)
        term2 = np.expand_dims(term2, 2)
        term3 = (norm_v3_inv + norm_v4_inv) / (norm_v3 * norm_v4 + v3_dot_v4)
        term3 = np.expand_dims(term3, 2)
        term4 = (norm_v4_inv + norm_v1_inv) / (norm_v4 * norm_v1 + v4_dot_v1)
        term4 = np.expand_dims(term4, 2)

        Vij_ring_vortices = 1 / (4 * np.pi) * (
                v1_cross_v2 * term1 +
                v2_cross_v3 * term2 +
                v3_cross_v4 * term3 +
                v4_cross_v1 * term4
        )

        return Vij_ring_vortices

    def calculate_Vij_horseshoe_vortices(self, points):
        # Calculates Vij, the velocity influence matrix (First index is collocation point number, second index is vortex
        # number).
        # points: the list of points (Nx3) to calculate the velocity influence at.

        # Make lv and rv
        left_vortex_vertices = self.left_horseshoe_vortex_vertices
        right_vortex_vertices = self.right_horseshoe_vortex_vertices

        points = np.reshape(points, (-1, 3))
        n_points = len(points)
        n_vortices = len(left_vortex_vertices)

        # Make a and b vectors.
        # a: Vector from all collocation points to all horseshoe vortex left  vertices, NxNx3.
        #   # First index is collocation point #, second is vortex #, and third is xyz. N=n_panels
        # b: Vector from all collocation points to all horseshoe vortex right vertices, NxNx3.
        #   # First index is collocation point #, second is vortex #, and third is xyz. N=n_panels
        # a[i,j,:] = c[i,:] - lv[j,:]
        # b[i,j,:] = c[i,:] - rv[j,:]
        points = np.expand_dims(points, 1)
        a = points - left_vortex_vertices
        b = points - right_vortex_vertices
        # x_hat = np.zeros([n_points, n_vortices, 3])
        # x_hat[:, :, 0] = 1

        # Do some useful arithmetic
        a_cross_b = np.cross(a, b, axis=2)
        a_dot_b = np.einsum('ijk,ijk->ij', a, b)

        a_cross_x = np.stack((
            np.zeros((n_points, n_vortices)),
            a[:, :, 2],
            -a[:, :, 1]
        ), axis=2)
        a_dot_x = a[:, :, 0]

        b_cross_x = np.stack((
            np.zeros((n_points, n_vortices)),
            b[:, :, 2],
            -b[:, :, 1]
        ), axis=2)
        b_dot_x = b[:, :, 0]  # np.sum(b * x_hat,axis=2)

        norm_a = np.linalg.norm(a, axis=2)
        norm_b = np.linalg.norm(b, axis=2)
        norm_a_inv = 1 / norm_a
        norm_b_inv = 1 / norm_b

        # Check for the special case where the collocation point is along the bound vortex leg
        # Find where cross product is near zero, and set the dot product to infinity so that the value of the bound term
        # is zero.
        bound_vortex_singularity_indices = (
                np.einsum('ijk,ijk->ij', a_cross_b, a_cross_b)  # norm(a_cross_b) ** 2
                < 3.0e-16)
        a_dot_b = a_dot_b + bound_vortex_singularity_indices
        left_vortex_singularity_indices = (
                np.einsum('ijk,ijk->ij', a_cross_x, a_cross_x)
                < 3.0e-16
        )
        a_dot_x = a_dot_x + left_vortex_singularity_indices
        right_vortex_singularity_indices = (
                np.einsum('ijk,ijk->ij', b_cross_x, b_cross_x)
                < 3.0e-16
        )
        b_dot_x = b_dot_x + right_vortex_singularity_indices

        # Calculate Vij
        term1 = (norm_a_inv + norm_b_inv) / (norm_a * norm_b + a_dot_b)
        term2 = norm_a_inv / (norm_a - a_dot_x)
        term3 = norm_b_inv / (norm_b - b_dot_x)
        term1 = np.expand_dims(term1, 2)
        term2 = np.expand_dims(term2, 2)
        term3 = np.expand_dims(term3, 2)

        Vij_horseshoe_vortices = 1 / (4 * np.pi) * (
                a_cross_b * term1 +
                a_cross_x * term2 -
                b_cross_x * term3
        )

        return Vij_horseshoe_vortices

    # ToDo Uncomment this function when the calculate_forces method is implemented
    """def calculate_delta_cp(self):
        # Find the area of each panel ()
        diag1 = self.front_left_vertices - self.back_right_vertices
        diag2 = self.front_right_vertices - self.back_left_vertices
        self.areas = np.linalg.norm(np.cross(diag1, diag2, axis=1), axis=1) / 2

        # Calculate panel data
        self.Fi_normal = np.einsum('ij,ij->i', self.Fi_geometry, self.normal_directions)
        self.pressure_normal = self.Fi_normal / self.areas
        self.delta_cp = self.pressure_normal / self.op_point.dynamic_pressure()"""

    def get_induced_velocity_at_point(self, point):
        # Input: a Nx3 numpy array of points that you would like to know the induced velocities at.
        # Output: a Nx3 numpy array of the induced velocities at those points.
        point = np.reshape(point, (-1, 3))

        Vij = self.calculate_Vij(point)

        vortex_strengths_expanded = np.expand_dims(self.vortex_strengths, 1)

        Vi_x = Vij[:, :, 0] @ vortex_strengths_expanded
        Vi_y = Vij[:, :, 1] @ vortex_strengths_expanded
        Vi_z = Vij[:, :, 2] @ vortex_strengths_expanded

        Vi = np.hstack((Vi_x, Vi_y, Vi_z))

        return Vi

    def get_velocity_at_point(self, point):
        # Input: a Nx3 numpy array of points that you would like to know the velocities at.
        # Output: a Nx3 numpy array of the velocities at those points.
        point = np.reshape(point, (-1, 3))

        Vi = self.get_induced_velocity_at_point(point)

        freestream = self.op_point.compute_freestream_velocity_geometry_axes()

        V = Vi + freestream

        return V

    def calculate_streamlines(self):
        # Calculates streamlines emanating from the trailing edges of all surfaces.
        # "streamlines" is a MxNx3 array, where M is the index of the streamline number,
        # N is the index of the timestep, and the last index is xyz

        # Constants
        n_steps = 100  # minimum of 2
        length = self.airplane.get_bounding_cube()[3]  # meter

        # Resolution
        length_per_step = length / n_steps

        # Seed points
        seed_points = (0.5 * (self.back_left_vertices + self.back_right_vertices))[self.is_trailing_edge]

        n_streamlines = len(seed_points)

        # Initialize
        streamlines = np.zeros((n_streamlines, n_steps, 3))
        streamlines[:, 0, :] = seed_points

        # Iterate
        for step_num in range(1, n_steps):
            update_amount = self.get_velocity_at_point(streamlines[:, step_num - 1, :])
            update_amount = update_amount * length_per_step / np.expand_dims(np.linalg.norm(update_amount, axis=1),
                                                                             axis=1)
            streamlines[:, step_num, :] = streamlines[:, step_num - 1, :] + update_amount

        self.streamlines = streamlines

    def draw(self, shading_type="ring_vortex_strengths",  # Can be None, "solid", or "ring_vortex_strengths"
             draw_streamlines=False,
             points_type=None,  # Supply the name of a property of this class to plot a point cloud
             ):  # Not autograd-compatible

        if self.verbose:
            print("Drawing...")

        # Initialize Plotter
        plotter = pv.Plotter()

        # ToDo Uncomment this code once the calculate_delta_cp and calculate_forces functions are complete
        """if draw_delta_cp:
            if not hasattr(self, 'delta_cp'):
                self.calculate_delta_cp()

            delta_cp_min = -1.5
            delta_cp_max = 1.5

            scalars = np.minimum(np.maximum(self.delta_cp, delta_cp_min), delta_cp_max)
            cmap = plt.cm.get_cmap('inferno')
            plotter.add_mesh(wing_surfaces, scalars=scalars, cmap=cmap, color='tan', show_edges=True,
                             smooth_shading=True)
            plotter.add_scalar_bar(title="Pressure Coefficient Differential", n_labels=5, shadow=True,
                                   font_family='arial')"""

        # Shading
        if shading_type is not None:

            # Make airplane geometry
            vertices = np.vstack((
                self.front_left_vertices,
                self.front_right_vertices,
                self.back_right_vertices,
                self.back_left_vertices
            ))
            faces = np.transpose(np.vstack((
                4 * np.ones(self.n_panels),
                np.arange(self.n_panels),
                np.arange(self.n_panels) + self.n_panels,
                np.arange(self.n_panels) + 2 * self.n_panels,
                np.arange(self.n_panels) + 3 * self.n_panels,
            )))
            faces = np.reshape(faces, (-1), order='C')
            wing_surfaces = pv.PolyData(vertices, faces)

            if shading_type == "solid":
                plotter.add_mesh(wing_surfaces, color='tan', show_edges=True,
                                 smooth_shading=True)
            else:
                if not hasattr(self, 'ring_vortex_strengths'):
                    print("Ring vortex strengths not found, running again.")
                    self.run()

                scalars = self.ring_vortex_strengths

                cmap = plt.cm.get_cmap('inferno')
                plotter.add_mesh(wing_surfaces, scalars=scalars, cmap=cmap, color='tan', show_edges=True,
                                 smooth_shading=True)
                plotter.add_scalar_bar(title="Ring Vortex Strengths", n_labels=5, shadow=True,
                                       font_family='arial')

        # Streamlines
        if draw_streamlines:
            if not hasattr(self, 'streamlines'):
                self.calculate_streamlines()

            for streamline_num in range(len(self.streamlines)):
                plotter.add_lines(self.streamlines[streamline_num, :, :], width=1, color='#50C7C7')

        # Points
        if points_type is not None:
            points = getattr(self, points_type)

            plotter.add_points(points)

        # Do the plotting
        plotter.show_grid(color='#444444')
        plotter.set_background(color="black")
        plotter.show(cpos=(-1, -1, 1), full_screen=False)

        if self.verbose:
            print("Drawing finished!")
