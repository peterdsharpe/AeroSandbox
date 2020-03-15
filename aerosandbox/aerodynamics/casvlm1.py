from .aerodynamics import *
from ..geometry import *


class Casvlm1(AeroProblem):
    # Usage:
    #   # Set up attrib_name problem using the syntax in the AeroProblem constructor (e.g. "Casvlm1(airplane = attrib_name, op_point = op)" for some Airplane attrib_name and OperatingPoint op)
    #   # Call the run() method on the vlm3 object to run the problem.
    #   # Access results in the command line, or through properties of the Casvlm1 class.
    #   #   # In attrib_name future update, this will be done through attrib_name standardized AeroData class.

    def __init__(self,
                 airplane,  # type: Airplane
                 op_point,  # type: op_point
                 opti,  # type: cas.Opti
                 run_setup = True,
                 ):
        super().__init__(airplane, op_point)
        self.opti = opti

        if run_setup:
            self.setup()

    def setup(self, verbose=True):
        # Runs attrib_name point analysis at the specified op-point.
        self.verbose = verbose

        if self.verbose: print("Setting up casVLM1 calculation...")

        self.make_panels()
        self.setup_geometry()
        self.setup_operating_point()
        self.calculate_vortex_strengths()
        self.calculate_forces()

        if self.verbose: print("casVLM1 setup complete! Ready to pass into the solver...")

    def make_panels(self):
        # Creates self.panel_coordinates_structured_list and self.wing_mcl_normals.

        if self.verbose: print("Meshing...")

        front_left_vertices = []
        front_right_vertices = []
        back_left_vertices = []
        back_right_vertices = []
        is_trailing_edge = []
        normal_directions = []

        for wing_num in range(len(self.airplane.wings)):
            # Things we want for each wing (where M is the number of chordwise panels, N is the number of spanwise panels)
            # # panel_coordinates_structured_list: M+1 p N+1 p 3; corners of every panel.
            # # normals_structured_list: M p N p 3; normal direction of each panel

            # Get the wing
            wing = self.airplane.wings[wing_num]  # type: Wing

            # Define number of chordwise points
            n_chordwise_coordinates = wing.chordwise_panels + 1

            # Get the chordwise coordinates
            if wing.chordwise_spacing == 'uniform':
                nondim_chordwise_coordinates = np.linspace(0, 1, n_chordwise_coordinates)
            elif wing.chordwise_spacing == 'cosine':
                nondim_chordwise_coordinates = np_cosspace(0, 1, n_chordwise_coordinates)
            else:
                raise Exception("Bad init_val of wing.chordwise_spacing!")

            # -----------------------------------------------------
            ## Make the panels for each section.

            for section_num in range(len(wing.xsecs) - 1):
                # Define the relevant cross sections
                inner_xsec = wing.xsecs[section_num]  # type: WingXSec
                outer_xsec = wing.xsecs[section_num + 1]  # type: WingXSec

                # Find the corners
                inner_xsec_xyz_le = inner_xsec.xyz_le + wing.xyz_le
                inner_xsec_xyz_te = inner_xsec.xyz_te() + wing.xyz_le
                outer_xsec_xyz_le = outer_xsec.xyz_le + wing.xyz_le
                outer_xsec_xyz_te = outer_xsec.xyz_te() + wing.xyz_le

                # Define the airfoils at each cross section
                inner_airfoil = inner_xsec.airfoil.get_airfoil_with_control_surface(
                    deflection=inner_xsec.control_surface_deflection,
                    hinge_point=inner_xsec.control_surface_hinge_point
                )  # type: Airfoil
                outer_airfoil = outer_xsec.airfoil.get_airfoil_with_control_surface(
                    deflection=inner_xsec.control_surface_deflection,
                    hinge_point=inner_xsec.control_surface_hinge_point
                )  # type: Airfoil

                # Make the mean camber lines for each.
                inner_xsec_mcl_nondim = inner_airfoil.get_downsampled_mcl(nondim_chordwise_coordinates)
                outer_xsec_mcl_nondim = outer_airfoil.get_downsampled_mcl(nondim_chordwise_coordinates)

                # Find the tangent angles of the mean camber lines
                inner_xsec_mcl_angle = (
                        cas.atan2(cas.diff(inner_xsec_mcl_nondim[:, 1]), cas.diff(inner_xsec_mcl_nondim[:, 0])) - (
                        inner_xsec.twist * np.pi / 180)
                )  # in radians
                outer_xsec_mcl_angle = (
                        cas.atan2(cas.diff(outer_xsec_mcl_nondim[:, 1]), cas.diff(outer_xsec_mcl_nondim[:, 0])) - (
                        outer_xsec.twist * np.pi / 180)
                )  # in radians

                # Find the effective twist axis
                effective_twist_axis = outer_xsec_xyz_le - inner_xsec_xyz_le
                effective_twist_axis[0] = 0.

                # Define number of spanwise points
                n_spanwise_coordinates = inner_xsec.spanwise_panels + 1

                # Get the spanwise coordinates
                if inner_xsec.spanwise_spacing == 'uniform':
                    nondim_spanwise_coordinates = np.linspace(0, 1, n_spanwise_coordinates)
                elif inner_xsec.spanwise_spacing == 'cosine':
                    nondim_spanwise_coordinates = np_cosspace(0, 1, n_spanwise_coordinates)
                else:
                    raise Exception("Bad init_val of section.spanwise_spacing!")

                for chord_index in range(wing.chordwise_panels):
                    for span_index in range(inner_xsec.spanwise_panels):
                        nondim_chordwise_coordinate = nondim_chordwise_coordinates[chord_index]
                        nondim_spanwise_coordinate = nondim_spanwise_coordinates[span_index]
                        nondim_chordwise_coordinate_next = nondim_chordwise_coordinates[chord_index + 1]
                        nondim_spanwise_coordinate_next = nondim_spanwise_coordinates[span_index + 1]

                        # Calculate vertices
                        front_left_vertices.append(
                            (
                                    inner_xsec_xyz_le * (
                                    1 - nondim_chordwise_coordinate) + inner_xsec_xyz_te * nondim_chordwise_coordinate
                            ) * (1 - nondim_spanwise_coordinate) +
                            (
                                    outer_xsec_xyz_le * (
                                    1 - nondim_chordwise_coordinate) + outer_xsec_xyz_te * nondim_chordwise_coordinate
                            ) * nondim_spanwise_coordinate
                        )
                        front_right_vertices.append(
                            (
                                    inner_xsec_xyz_le * (
                                    1 - nondim_chordwise_coordinate) + inner_xsec_xyz_te * nondim_chordwise_coordinate
                            ) * (1 - nondim_spanwise_coordinate_next) +
                            (
                                    outer_xsec_xyz_le * (
                                    1 - nondim_chordwise_coordinate) + outer_xsec_xyz_te * nondim_chordwise_coordinate
                            ) * nondim_spanwise_coordinate_next
                        )
                        back_left_vertices.append(
                            (
                                    inner_xsec_xyz_le * (
                                    1 - nondim_chordwise_coordinate_next) + inner_xsec_xyz_te * nondim_chordwise_coordinate_next
                            ) * (1 - nondim_spanwise_coordinate) +
                            (
                                    outer_xsec_xyz_le * (
                                    1 - nondim_chordwise_coordinate_next) + outer_xsec_xyz_te * nondim_chordwise_coordinate_next
                            ) * nondim_spanwise_coordinate
                        )
                        back_right_vertices.append(
                            (
                                    inner_xsec_xyz_le * (
                                    1 - nondim_chordwise_coordinate_next) + inner_xsec_xyz_te * nondim_chordwise_coordinate_next
                            ) * (1 - nondim_spanwise_coordinate_next) +
                            (
                                    outer_xsec_xyz_le * (
                                    1 - nondim_chordwise_coordinate_next) + outer_xsec_xyz_te * nondim_chordwise_coordinate_next
                            ) * nondim_spanwise_coordinate_next
                        )
                        is_trailing_edge.append(chord_index == wing.chordwise_panels - 1)

                        # Calculate normal
                        angle = (
                                inner_xsec_mcl_angle[chord_index] * (1 - nondim_spanwise_coordinate) +
                                inner_xsec_mcl_angle[chord_index] * nondim_spanwise_coordinate
                        )
                        rot = angle_axis_rotation_matrix(-angle - np.pi / 2, effective_twist_axis)
                        normal_directions.append(rot @ cas.vertcat(1, 0, 0))

                # Handle symmetry
                if wing.symmetric:
                    # Define the relevant cross sections
                    inner_xsec = wing.xsecs[section_num]  # type: WingXSec
                    outer_xsec = wing.xsecs[section_num + 1]  # type: WingXSec

                    # Define the airfoils at each cross section
                    if inner_xsec.control_surface_type == "symmetric":
                        inner_airfoil = inner_xsec.airfoil.get_airfoil_with_control_surface(
                            deflection=inner_xsec.control_surface_deflection,
                            hinge_point=inner_xsec.control_surface_hinge_point
                        )  # type: Airfoil
                        outer_airfoil = outer_xsec.airfoil.get_airfoil_with_control_surface(
                            deflection=inner_xsec.control_surface_deflection,
                            hinge_point=inner_xsec.control_surface_hinge_point
                        )  # type: Airfoil
                    elif inner_xsec.control_surface_type == "asymmetric":
                        inner_airfoil = inner_xsec.airfoil.get_airfoil_with_control_surface(
                            deflection=-inner_xsec.control_surface_deflection,
                            hinge_point=inner_xsec.control_surface_hinge_point
                        )  # type: Airfoil
                        outer_airfoil = outer_xsec.airfoil.get_airfoil_with_control_surface(
                            deflection=-inner_xsec.control_surface_deflection,
                            hinge_point=inner_xsec.control_surface_hinge_point
                        )  # type: Airfoil
                    else:
                        raise ValueError("Invalid input for control_surface_type!")

                    # Make the mean camber lines for each.
                    inner_xsec_mcl_nondim = inner_airfoil.get_downsampled_mcl(nondim_chordwise_coordinates)
                    outer_xsec_mcl_nondim = outer_airfoil.get_downsampled_mcl(nondim_chordwise_coordinates)

                    # Find the tangent angles of the mean camber lines
                    inner_xsec_mcl_angle = (
                            cas.atan2(cas.diff(inner_xsec_mcl_nondim[:, 1]),
                                      cas.diff(inner_xsec_mcl_nondim[:, 0])) - (
                                    inner_xsec.twist * np.pi / 180)
                    )  # in radians
                    outer_xsec_mcl_angle = (
                            cas.atan2(cas.diff(outer_xsec_mcl_nondim[:, 1]),
                                      cas.diff(outer_xsec_mcl_nondim[:, 0])) - (
                                    outer_xsec.twist * np.pi / 180)
                    )  # in radians

                    # Find the effective twist axis
                    effective_twist_axis = outer_xsec_xyz_le - inner_xsec_xyz_le
                    effective_twist_axis[0] = 0.

                    # Define number of spanwise points
                    n_spanwise_coordinates = inner_xsec.spanwise_panels + 1

                    # Get the spanwise coordinates
                    if inner_xsec.spanwise_spacing == 'uniform':
                        nondim_spanwise_coordinates = np.linspace(0, 1, n_spanwise_coordinates)
                    elif inner_xsec.spanwise_spacing == 'cosine':
                        nondim_spanwise_coordinates = np_cosspace(0, 1, n_spanwise_coordinates)
                    else:
                        raise Exception("Bad init_val of section.spanwise_spacing!")

                    for chord_index in range(wing.chordwise_panels):
                        for span_index in range(inner_xsec.spanwise_panels):
                            nondim_chordwise_coordinate = nondim_chordwise_coordinates[chord_index]
                            nondim_spanwise_coordinate = nondim_spanwise_coordinates[span_index]
                            nondim_chordwise_coordinate_next = nondim_chordwise_coordinates[chord_index + 1]
                            nondim_spanwise_coordinate_next = nondim_spanwise_coordinates[span_index + 1]

                            # Calculate vertices
                            front_right_vertices.append(reflect_over_XZ_plane(
                                (
                                        inner_xsec_xyz_le * (
                                        1 - nondim_chordwise_coordinate) + inner_xsec_xyz_te * nondim_chordwise_coordinate
                                ) * (1 - nondim_spanwise_coordinate) +
                                (
                                        outer_xsec_xyz_le * (
                                        1 - nondim_chordwise_coordinate) + outer_xsec_xyz_te * nondim_chordwise_coordinate
                                ) * nondim_spanwise_coordinate
                            ))
                            front_left_vertices.append(reflect_over_XZ_plane(
                                (
                                        inner_xsec_xyz_le * (
                                        1 - nondim_chordwise_coordinate) + inner_xsec_xyz_te * nondim_chordwise_coordinate
                                ) * (1 - nondim_spanwise_coordinate_next) +
                                (
                                        outer_xsec_xyz_le * (
                                        1 - nondim_chordwise_coordinate) + outer_xsec_xyz_te * nondim_chordwise_coordinate
                                ) * nondim_spanwise_coordinate_next
                            ))
                            back_right_vertices.append(reflect_over_XZ_plane(
                                (
                                        inner_xsec_xyz_le * (
                                        1 - nondim_chordwise_coordinate_next) + inner_xsec_xyz_te * nondim_chordwise_coordinate_next
                                ) * (1 - nondim_spanwise_coordinate) +
                                (
                                        outer_xsec_xyz_le * (
                                        1 - nondim_chordwise_coordinate_next) + outer_xsec_xyz_te * nondim_chordwise_coordinate_next
                                ) * nondim_spanwise_coordinate
                            ))
                            back_left_vertices.append(reflect_over_XZ_plane(
                                (
                                        inner_xsec_xyz_le * (
                                        1 - nondim_chordwise_coordinate_next) + inner_xsec_xyz_te * nondim_chordwise_coordinate_next
                                ) * (1 - nondim_spanwise_coordinate_next) +
                                (
                                        outer_xsec_xyz_le * (
                                        1 - nondim_chordwise_coordinate_next) + outer_xsec_xyz_te * nondim_chordwise_coordinate_next
                                ) * nondim_spanwise_coordinate_next
                            ))
                            is_trailing_edge.append(chord_index == wing.chordwise_panels - 1)

                            # Calculate normal
                            angle = (
                                    inner_xsec_mcl_angle[chord_index] * (1 - nondim_spanwise_coordinate) +
                                    inner_xsec_mcl_angle[chord_index] * nondim_spanwise_coordinate
                            )
                            rot = angle_axis_rotation_matrix(-angle - np.pi / 2, effective_twist_axis)
                            normal_directions.append(reflect_over_XZ_plane(rot @ cas.vertcat(1, 0, 0)))

        # Concatenate things (DM)
        # self.front_left_vertices = cas.transpose(cas.horzcat(*front_left_vertices))
        # self.front_right_vertices = cas.transpose(cas.horzcat(*front_right_vertices))
        # self.back_left_vertices = cas.transpose(cas.horzcat(*back_left_vertices))
        # self.back_right_vertices = cas.transpose(cas.horzcat(*back_right_vertices))
        # self.normal_directions = cas.transpose(cas.horzcat(*normal_directions))
        # self.is_trailing_edge = is_trailing_edge

        # Concatenate things (MX)
        self.front_left_vertices = cas.MX(cas.transpose(cas.horzcat(*front_left_vertices)))
        self.front_right_vertices = cas.MX(cas.transpose(cas.horzcat(*front_right_vertices)))
        self.back_left_vertices = cas.MX(cas.transpose(cas.horzcat(*back_left_vertices)))
        self.back_right_vertices = cas.MX(cas.transpose(cas.horzcat(*back_right_vertices)))
        self.normal_directions = cas.MX(cas.transpose(cas.horzcat(*normal_directions)))
        self.is_trailing_edge = is_trailing_edge

        # Calculate areas
        diag1 = self.front_right_vertices - self.back_left_vertices
        diag2 = self.front_left_vertices - self.back_right_vertices
        cross = cas.cross(diag1, diag2)
        cross_norm = cas.sqrt(cross[:, 0] ** 2 + cross[:, 1] ** 2 + cross[:, 2] ** 2)
        self.areas = cross_norm / 2

        # Do the vortex math
        self.left_vortex_vertices = 0.75 * self.front_left_vertices + 0.25 * self.back_left_vertices
        self.right_vortex_vertices = 0.75 * self.front_right_vertices + 0.25 * self.back_right_vertices
        self.vortex_centers = (self.left_vortex_vertices + self.right_vortex_vertices) / 2
        self.vortex_bound_leg = (self.right_vortex_vertices - self.left_vortex_vertices)
        self.collocation_points = (
                0.5 * (
                0.25 * self.front_left_vertices + 0.75 * self.back_left_vertices
        ) +
                0.5 * (
                        0.25 * self.front_right_vertices + 0.75 * self.back_right_vertices
                )
        )

        # Do final processing for later use
        self.n_panels = self.collocation_points.shape[0]

        if self.verbose: print("Meshing complete!")

    def setup_geometry(self):
        # # Calculate AIC matrix
        # ----------------------
        if self.verbose: print("Calculating the collocation influence matrix...")
        self.Vij_collocations_x, self.Vij_collocations_y, self.Vij_collocations_z = self.calculate_Vij(
            self.collocation_points)

        # AIC = (Vij * normal vectors)
        self.AIC = (
                self.Vij_collocations_x * self.normal_directions[:, 0] +
                self.Vij_collocations_y * self.normal_directions[:, 1] +
                self.Vij_collocations_z * self.normal_directions[:, 2]
        )

        # # Calculate Vij at vortex centers for force calculation
        # -------------------------------------------------------
        if self.verbose: print("Calculating the vortex center influence matrix...")
        self.Vij_centers_x, self.Vij_centers_y, self.Vij_centers_z = self.calculate_Vij(self.vortex_centers)

    def setup_operating_point(self):
        if self.verbose: print("Calculating the freestream influence...")
        self.steady_freestream_velocity = self.op_point.compute_freestream_velocity_geometry_axes()  # Direction the wind is GOING TO, in geometry axes coordinates
        self.rotation_freestream_velocities = self.op_point.compute_rotation_velocity_geometry_axes(
            self.collocation_points)

        self.freestream_velocities = cas.transpose(self.steady_freestream_velocity + cas.transpose(
            self.rotation_freestream_velocities))  # Nx3, represents the freestream velocity at each panel collocation point (c)

        self.freestream_influences = (
                self.freestream_velocities[:, 0] * self.normal_directions[:, 0] +
                self.freestream_velocities[:, 1] * self.normal_directions[:, 1] +
                self.freestream_velocities[:, 2] * self.normal_directions[:, 2]
        )

    def calculate_vortex_strengths(self):
        # # Calculate Vortex Strengths
        # ----------------------------
        # Governing Equation: AIC @ Gamma + freestream_influence = 0
        if self.verbose: print("Calculating vortex strengths...")

        # Explicit solve
        self.vortex_strengths = cas.solve(self.AIC, -self.freestream_influences)

        # # Implicit solve
        # self.vortex_strengths = self.opti.variable(self.n_panels)
        # self.opti.set_initial(self.vortex_strengths, 1)
        # self.opti.subject_to([
        #     self.AIC @ self.vortex_strengths == -self.freestream_influences
        # ])

    def calculate_forces(self):
        # # Calculate Near-Field Forces and Moments
        # -----------------------------------------
        # Governing Equation: The force on attrib_name straight, small vortex filament is F = rho * V p l * gamma,
        # where rho is density, V is the velocity vector, p is the cross product operator,
        # l is the vector of the filament itself, and gamma is the circulation.

        if self.verbose: print("Calculating forces on each panel...")
        # Calculate Vi (local velocity at the ith vortex center point)
        Vi_x = self.Vij_centers_x @ self.vortex_strengths + self.freestream_velocities[:, 0]
        Vi_y = self.Vij_centers_y @ self.vortex_strengths + self.freestream_velocities[:, 1]
        Vi_z = self.Vij_centers_z @ self.vortex_strengths + self.freestream_velocities[:, 2]
        Vi = cas.horzcat(Vi_x, Vi_y, Vi_z)

        # Calculate forces_inviscid_geometry, the force on the ith panel. Note that this is in GEOMETRY AXES,
        # not WIND AXES or BODY AXES.
        density = self.op_point.density
        # Vi_cross_li = np.cross(Vi, self.vortex_bound_leg, axis=1)
        Vi_cross_li = cas.horzcat(
            Vi_y * self.vortex_bound_leg[:, 2] - Vi_z * self.vortex_bound_leg[:, 1],
            Vi_z * self.vortex_bound_leg[:, 0] - Vi_x * self.vortex_bound_leg[:, 2],
            Vi_x * self.vortex_bound_leg[:, 1] - Vi_y * self.vortex_bound_leg[:, 0],
        )
        # vortex_strengths_expanded = np.expand_dims(self.vortex_strengths, axis=1)
        self.Fi_geometry = density * Vi_cross_li * self.vortex_strengths

        # Calculate total forces and moments
        if self.verbose: print("Calculating total forces and moments...")
        self.Ftotal_geometry = cas.vertcat(
            cas.sum1(self.Fi_geometry[:, 0]),
            cas.sum1(self.Fi_geometry[:, 1]),
            cas.sum1(self.Fi_geometry[:, 2]),
        )  # Remember, this is in GEOMETRY AXES, not WIND AXES or BODY AXES.
        # if self.verbose: print("Total aerodynamic forces (geometry axes): ", self.force_total_inviscid_geometry)

        self.Ftotal_wind = cas.transpose(
            self.op_point.compute_rotation_matrix_wind_to_geometry()) @ self.Ftotal_geometry
        # if self.verbose: print("Total aerodynamic forces (wind axes):", self.force_total_inviscid_wind)

        self.Mi_geometry = cas.cross(
            cas.transpose(cas.transpose(self.vortex_centers) - self.airplane.xyz_ref),
            self.Fi_geometry
        )

        self.Mtotal_geometry = cas.vertcat(
            cas.sum1(self.Mi_geometry[:, 0]),
            cas.sum1(self.Mi_geometry[:, 1]),
            cas.sum1(self.Mi_geometry[:, 2]),
        )

        self.Mtotal_wind = cas.transpose(
            self.op_point.compute_rotation_matrix_wind_to_geometry()) @ self.Mtotal_geometry

        # Calculate nondimensional forces
        q = self.op_point.dynamic_pressure()
        s_ref = self.airplane.s_ref
        b_ref = self.airplane.b_ref
        c_ref = self.airplane.c_ref
        self.CL = -self.Ftotal_wind[2] / q / s_ref
        self.CDi = -self.Ftotal_wind[0] / q / s_ref
        self.CY = self.Ftotal_wind[1] / q / s_ref
        self.Cl = self.Mtotal_wind[0] / q / s_ref / b_ref
        self.Cm = self.Mtotal_wind[1] / q / s_ref / c_ref
        self.Cn = self.Mtotal_wind[2] / q / s_ref / b_ref

        # Solves divide by zero error
        self.CL_over_CDi = cas.if_else(self.CDi == 0, 0, self.CL / self.CDi)

    def calculate_Vij(self, points):
        # Calculates Vij, the velocity influence matrix (First index is collocation point number, second index is vortex number).
        # points: the list of points (Nx3) to calculate the velocity influence at.

        # Make lv and rv
        left_vortex_vertices = self.left_vortex_vertices
        right_vortex_vertices = self.right_vortex_vertices

        n_points = points.shape[0]
        n_vortices = self.n_panels

        # Make attrib_name and b vectors.
        # attrib_name: Vector from all collocation points to all horseshoe vortex left  vertices, NxNx3.
        #   # First index is collocation point #, second is vortex #, and third is xyz. N=n_panels
        # b: Vector from all collocation points to all horseshoe vortex right vertices, NxNx3.
        #   # First index is collocation point #, second is vortex #, and third is xyz. N=n_panels
        # attrib_name[i,j,:] = c[i,:] - lv[j,:]
        # b[i,j,:] = c[i,:] - rv[j,:]
        a_x = points[:, 0] - cas.repmat(cas.transpose(left_vortex_vertices[:, 0]), n_points, 1)
        a_y = points[:, 1] - cas.repmat(cas.transpose(left_vortex_vertices[:, 1]), n_points, 1)
        a_z = points[:, 2] - cas.repmat(cas.transpose(left_vortex_vertices[:, 2]), n_points, 1)
        b_x = points[:, 0] - cas.repmat(cas.transpose(right_vortex_vertices[:, 0]), n_points, 1)
        b_y = points[:, 1] - cas.repmat(cas.transpose(right_vortex_vertices[:, 1]), n_points, 1)
        b_z = points[:, 2] - cas.repmat(cas.transpose(right_vortex_vertices[:, 2]), n_points, 1)

        # x_hat = np.zeros([n_points, n_vortices, 3])
        # x_hat[:, :, 0] = 1

        # Do some useful arithmetic
        a_cross_b_x = a_y * b_z - a_z * b_y
        a_cross_b_y = a_z * b_x - a_x * b_z
        a_cross_b_z = a_x * b_y - a_y * b_x
        a_dot_b = a_x * b_x + a_y * b_y + a_z * b_z

        a_cross_xhat_x = 0
        a_cross_xhat_y = a_z
        a_cross_xhat_z = -a_y
        a_dot_xhat = a_x

        b_cross_xhat_x = 0
        b_cross_xhat_y = b_z
        b_cross_xhat_z = -b_y
        b_dot_xhat = b_x

        norm_a = cas.sqrt(a_x ** 2 + a_y ** 2 + a_z ** 2)
        norm_b = cas.sqrt(b_x ** 2 + b_y ** 2 + b_z ** 2)
        norm_a_inv = 1 / norm_a
        norm_b_inv = 1 / norm_b

        # Handle the special case where the collocation point is along the bound vortex leg
        a_dot_b -= 1e-8
        # a_dot_xhat += 1e-8
        # b_dot_xhat += 1e-8

        # Calculate Vij
        term1 = (norm_a_inv + norm_b_inv) / (norm_a * norm_b + a_dot_b)
        term2 = (norm_a_inv) / (norm_a - a_dot_xhat)
        term3 = (norm_b_inv) / (norm_b - b_dot_xhat)

        Vij_x = 1 / (4 * np.pi) * (
                a_cross_b_x * term1 +
                a_cross_xhat_x * term2 -
                b_cross_xhat_x * term3
        )
        Vij_y = 1 / (4 * np.pi) * (
                a_cross_b_y * term1 +
                a_cross_xhat_y * term2 -
                b_cross_xhat_y * term3
        )
        Vij_z = 1 / (4 * np.pi) * (
                a_cross_b_z * term1 +
                a_cross_xhat_z * term2 -
                b_cross_xhat_z * term3
        )

        return Vij_x, Vij_y, Vij_z

    # def calculate_delta_cp(self):
    #     # Find the area of each panel ()
    #     diag1 = self.front_left_vertices - self.back_right_vertices
    #     diag2 = self.front_right_vertices - self.back_left_vertices
    #     self.areas = np.linalg.norm(np.cross(diag1, diag2, axis=1), axis=1) / 2
    #
    #     # Calculate panel data
    #     self.Fi_normal = np.einsum('ij,ij->i', self.forces_inviscid_geometry, self.normal_directions)
    #     self.pressure_normal = self.Fi_normal / self.areas
    #     self.delta_cp = self.pressure_normal / self.op_point.dynamic_pressure()

    def get_induced_velocity_at_point(self, point):
        if not self.opti.return_status() == 'Solve_Succeeded':
            print("WARNING: This method should only be used after a solution has been found!!!\n"
                  "Running anyway for debugging purposes - this is likely to not work.")

        Vij_x, Vij_y, Vij_z = self.calculate_Vij(point)

        vortex_strengths = self.opti.debug.value(self.vortex_strengths)

        Vi_x = Vij_x @ vortex_strengths
        Vi_y = Vij_y @ vortex_strengths
        Vi_z = Vij_z @ vortex_strengths

        Vi = np.hstack((Vi_x, Vi_y, Vi_z))

        return Vi

    def get_velocity_at_point(self, point):
        # Input: attrib_name Nx3 numpy array of points that you would like to know the velocities at.
        # Output: attrib_name Nx3 numpy array of the velocities at those points.

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

        n_streamlines = len(seed_points)

        # Initialize
        streamlines = []
        streamlines.append(seed_points)

        # Iterate
        for step_num in range(1, n_steps):
            update_amount = self.get_velocity_at_point(streamlines[-1])
            norm_update_amount = cas.sqrt(
                update_amount[:, 0] ** 2 + update_amount[:, 1] ** 2 + update_amount[:, 2] ** 2)
            update_amount = length_per_step * update_amount / norm_update_amount
            streamlines.append(streamlines[-1] + update_amount)

        self.streamlines = streamlines



    def draw(self, data_to_plot=None, data_name=None, show=True, draw_streamlines=True, recalculate_streamlines=False):
        """
        Draws the solution. Note: Must be called on a SOLVED AeroProblem object.
        To solve an AeroProblem, use opti.solve(). To substitute a solved solution, use ap = ap.substitute_solution(sol).
        :return:
        """
        print("Drawing...")

        if not self.opti.return_status() == 'Solve_Succeeded':
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
        try:
            data_to_plot = get(data_to_plot)
        except NotImplementedError:
            pass

        if data_to_plot is None:
            data_name = "Vortex Strengths"
            data_to_plot = get(self.vortex_strengths)

        fig = go.Figure()

        # x, y, and z give the vertices
        x = []
        y = []
        z = []
        # i, j and k give the connectivity of the vertices
        i = []
        j = []
        k = []
        intensity = []
        # xe, ye, and ze give the outline of each panel
        xe = []
        ye = []
        ze = []

        for index in range(len(front_left_vertices)):
            x.append(front_left_vertices[index, 0])
            x.append(front_right_vertices[index, 0])
            x.append(back_right_vertices[index, 0])
            x.append(back_left_vertices[index, 0])
            y.append(front_left_vertices[index, 1])
            y.append(front_right_vertices[index, 1])
            y.append(back_right_vertices[index, 1])
            y.append(back_left_vertices[index, 1])
            z.append(front_left_vertices[index, 2])
            z.append(front_right_vertices[index, 2])
            z.append(back_right_vertices[index, 2])
            z.append(back_left_vertices[index, 2])
            intensity.append(data_to_plot[index])
            intensity.append(data_to_plot[index])
            intensity.append(data_to_plot[index])
            intensity.append(data_to_plot[index])
            xe.append(front_left_vertices[index, 0])
            xe.append(front_right_vertices[index, 0])
            xe.append(back_right_vertices[index, 0])
            xe.append(back_left_vertices[index, 0])
            ye.append(front_left_vertices[index, 1])
            ye.append(front_right_vertices[index, 1])
            ye.append(back_right_vertices[index, 1])
            ye.append(back_left_vertices[index, 1])
            ze.append(front_left_vertices[index, 2])
            ze.append(front_right_vertices[index, 2])
            ze.append(back_right_vertices[index, 2])
            ze.append(back_left_vertices[index, 2])
            xe.append(None)
            ye.append(None)
            ze.append(None)
            xe.append(left_vortex_vertices[index, 0])
            xe.append(right_vortex_vertices[index, 0])
            ye.append(left_vortex_vertices[index, 1])
            ye.append(right_vortex_vertices[index, 1])
            ze.append(left_vortex_vertices[index, 2])
            ze.append(right_vortex_vertices[index, 2])
            xe.append(None)
            ye.append(None)
            ze.append(None)

            indices_added = np.arange(len(x) - 4, len(x))

            # Add front_left triangle
            i.append(indices_added[0])
            j.append(indices_added[1])
            k.append(indices_added[3])
            # Add back-right triangle
            i.append(indices_added[2])
            j.append(indices_added[3])
            k.append(indices_added[1])

            # if self.symmetric_problem:
            #     if self.use_symmetry[index]:
            #         x.append(front_left_vertices[index, 0])
            #         x.append(front_right_vertices[index, 0])
            #         x.append(back_right_vertices[index, 0])
            #         x.append(back_left_vertices[index, 0])
            #         y.append(-front_left_vertices[index, 1])
            #         y.append(-front_right_vertices[index, 1])
            #         y.append(-back_right_vertices[index, 1])
            #         y.append(-back_left_vertices[index, 1])
            #         z.append(front_left_vertices[index, 2])
            #         z.append(front_right_vertices[index, 2])
            #         z.append(back_right_vertices[index, 2])
            #         z.append(back_left_vertices[index, 2])
            #         intensity.append(data_to_plot[index])
            #         intensity.append(data_to_plot[index])
            #         intensity.append(data_to_plot[index])
            #         intensity.append(data_to_plot[index])
            #         xe.append(front_left_vertices[index, 0])
            #         xe.append(front_right_vertices[index, 0])
            #         xe.append(back_right_vertices[index, 0])
            #         xe.append(back_left_vertices[index, 0])
            #         ye.append(-front_left_vertices[index, 1])
            #         ye.append(-front_right_vertices[index, 1])
            #         ye.append(-back_right_vertices[index, 1])
            #         ye.append(-back_left_vertices[index, 1])
            #         ze.append(front_left_vertices[index, 2])
            #         ze.append(front_right_vertices[index, 2])
            #         ze.append(back_right_vertices[index, 2])
            #         ze.append(back_left_vertices[index, 2])
            #         xe.append(None)
            #         ye.append(None)
            #         ze.append(None)
            #         xe.append(left_vortex_vertices[index, 0])
            #         xe.append(right_vortex_vertices[index, 0])
            #         ye.append(-left_vortex_vertices[index, 1])
            #         ye.append(-right_vortex_vertices[index, 1])
            #         ze.append(left_vortex_vertices[index, 2])
            #         ze.append(right_vortex_vertices[index, 2])
            #         xe.append(None)
            #         ye.append(None)
            #         ze.append(None)
            #
            #         indices_added = np.arange(len(x) - 4, len(x))
            #
            #         # Add front_left triangle
            #         i.append(indices_added[0])
            #         j.append(indices_added[1])
            #         k.append(indices_added[3])
            #         # Add back-right triangle
            #         i.append(indices_added[2])
            #         j.append(indices_added[3])
            #         k.append(indices_added[1])

        fig.add_trace(
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                flatshading=False,
                intensity=intensity,
                colorscale="Viridis",
                colorbar=dict(
                    title=data_name,
                    titleside="top",
                    ticks="outside"
                )
            )
        )

        # define the trace for triangle sides
        fig.add_trace(
            go.Scatter3d(
                x=xe,
                y=ye,
                z=ze,
                mode='lines',
                name='',
                line=dict(color='rgb(0,0,0)', width=2),
                showlegend=False
            )
        )

        if draw_streamlines:
            if (not hasattr(self, 'streamlines')) or recalculate_streamlines:
                if self.verbose: print("Calculating streamlines...")
                back_centers = (self.back_left_vertices + self.back_right_vertices) / 2
                seed_points = []
                for index in range(back_centers.shape[0]):
                    if self.is_trailing_edge[index]:
                        seed_points.append(back_centers[index,:])
                seed_points = np.vstack(seed_points)

                self.calculate_streamlines(seed_points=seed_points)

            if self.verbose: print("Parsing streamline data...")
            n_streamlines = self.streamlines[0].shape[0]
            n_timesteps = len(self.streamlines)

            xs = []
            ys = []
            zs = []

            for streamlines_num in range(n_streamlines):
                xs.extend([float(self.streamlines[ts][streamlines_num, 0]) for ts in range(n_timesteps)])
                ys.extend([float(self.streamlines[ts][streamlines_num, 1]) for ts in range(n_timesteps)])
                zs.extend([float(self.streamlines[ts][streamlines_num, 2]) for ts in range(n_timesteps)])

                xs.append(None)
                ys.append(None)
                zs.append(None)

                # if self.symmetric_problem:  # TODO consider removing redundant plotting of centerline surfaces (low priority)
                #     xs.extend([float(self.streamlines[ts][streamlines_num, 0]) for ts in range(n_timesteps)])
                #     ys.extend([-float(self.streamlines[ts][streamlines_num, 1]) for ts in range(n_timesteps)])
                #     zs.extend([float(self.streamlines[ts][streamlines_num, 2]) for ts in range(n_timesteps)])
                #
                #     xs.append(None)
                #     ys.append(None)
                #     zs.append(None)

            fig.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode='lines',
                    name='',
                    line=dict(color='#7700ff', width=1),
                    showlegend=False
                )
            )

        fig.update_layout(
            title="%s Airplane, CasVLM1 Solution" % self.airplane.name,
            scene=dict(aspectmode='data'),

        )

        if show: fig.show()

        return fig