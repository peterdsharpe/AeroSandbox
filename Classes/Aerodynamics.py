import numpy as np
import scipy.linalg as sp_linalg
import scipy.sparse as sp_sparse
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from .Plotting import *
from .Geometry import *
from .Performance import *


class AeroProblem:
    def __init__(self,
                 airplane=Airplane(),
                 op_point=OperatingPoint(),
                 panels=[]
                 ):
        self.airplane = airplane
        self.op_point = op_point
        self.panels = panels
        self.problem_type = None

    def make_vlm1_problem(self):
        # Traditional Vortex Lattice Method approach with quadrilateral paneling, horseshoe vortices from each one, etc.
        # Implemented exactly as The Good Book says (Drela, "Flight Vehicle Aerodynamics", p. 130-135)
        self.problem_type = 'VLM1'
        print("Running the VLM1 calculation:")

        # # Make panels
        # -------------

        print("Making panels...")
        for wing in self.airplane.wings:
            for section_num in range(0, len(wing.sections) - 1):

                # Define the relevant sections
                section = wing.sections[section_num]
                next_section = wing.sections[section_num + 1]

                # Define number of chordwise and spanwise points
                n_chordwise_coordinates = section.chordwise_panels + 1
                n_spanwise_coordinates = section.spanwise_panels + 1

                # Get the chordwise coordinates
                if section.chordwise_spacing == 'uniform':
                    nondim_chordwise_coordinates = np.linspace(0, 1, n_chordwise_coordinates)
                elif section.chordwise_spacing == 'cosine':
                    nondim_chordwise_coordinates = 0.5 + 0.5 * np.cos(np.linspace(np.pi, 0, n_chordwise_coordinates))
                else:
                    raise Exception("Bad value of section.chordwise_spacing!")

                # Get the spanwise coordinates
                if section.spanwise_spacing == 'uniform':
                    nondim_spanwise_coordinates = np.linspace(0, 1, n_spanwise_coordinates)
                elif section.spanwise_spacing == 'cosine':
                    nondim_spanwise_coordinates = 0.5 + 0.5 * np.cos(np.linspace(np.pi, 0, n_spanwise_coordinates))
                else:
                    raise Exception("Bad value of section.spanwise_spacing!")

                # Get edges of WingSection (needed for the next step)
                section_xyz_le = section.xyz_le + wing.xyz_le
                section_xyz_te = section.xyz_te() + wing.xyz_le
                next_section_xyz_le = next_section.xyz_le + wing.xyz_le
                next_section_xyz_te = next_section.xyz_te() + wing.xyz_le

                # # For debugging
                # print("")
                # print(wing.name, ", Section ", section_num)
                # print("section_xyz_le ", section_xyz_le)
                # print("section_xyz_te ", section_xyz_te)
                # print("next_section_xyz_le ", next_section_xyz_le)
                # print("next_section_xyz_te ", next_section_xyz_te)

                # Initialize an array of coordinates. Indices:
                #   Index 1: chordwise location
                #   Index 2: spanwise location
                #   Index 3: X, Y, or Z.
                coordinates = np.zeros(shape=(n_chordwise_coordinates, n_spanwise_coordinates, 3))

                # Dimensionalize the chordwise and spanwise coordinates
                for spanwise_coordinate_num in range(len(nondim_spanwise_coordinates)):
                    nondim_spanwise_coordinate = nondim_spanwise_coordinates[spanwise_coordinate_num]

                    local_xyz_le = ((1 - nondim_spanwise_coordinate) * section_xyz_le +
                                    (nondim_spanwise_coordinate) * next_section_xyz_le)
                    local_xyz_te = ((1 - nondim_spanwise_coordinate) * section_xyz_te +
                                    (nondim_spanwise_coordinate) * next_section_xyz_te)

                    for chordwise_coordinate_num in range(len(nondim_chordwise_coordinates)):
                        nondim_chordwise_coordinate = nondim_chordwise_coordinates[chordwise_coordinate_num]

                        local_coordinate = ((1 - nondim_chordwise_coordinate) * local_xyz_le +
                                            (nondim_chordwise_coordinate) * local_xyz_te)

                        coordinates[chordwise_coordinate_num, spanwise_coordinate_num, :] = local_coordinate

                x = coordinates[:, :, 0]
                y = coordinates[:, :, 1]
                z = coordinates[:, :, 2]

                # # For debugging: view the coordinates of each WingSection
                # fig,ax = fig3d()
                # ax.scatter(x,y,z,c='b',marker='.')
                # ax.set_xlabel("X")
                # ax.set_ylabel("Y")
                # ax.set_zlabel("Z")
                # set_axes_equal(ax)
                # plt.show()

                # Make the panels
                for spanwise_coordinate_num in range(len(nondim_spanwise_coordinates) - 1):
                    for chordwise_coordinate_num in range(len(nondim_chordwise_coordinates) - 1):

                        front_inboard = coordinates[chordwise_coordinate_num, spanwise_coordinate_num, :]
                        front_outboard = coordinates[chordwise_coordinate_num, spanwise_coordinate_num + 1, :]
                        back_inboard = coordinates[chordwise_coordinate_num + 1, spanwise_coordinate_num, :]
                        back_outboard = coordinates[chordwise_coordinate_num + 1, spanwise_coordinate_num + 1, :]

                        vertices = np.vstack((
                            front_inboard,
                            front_outboard,
                            back_outboard,
                            back_inboard
                        ))

                        colocation_point = (
                                0.25 * np.mean(np.vstack((front_inboard, front_outboard)), axis=0) +
                                0.75 * np.mean(np.vstack((back_inboard, back_outboard)), axis=0)
                        )

                        # Compute normal vector by normalizing the cross product of two diagonals # TODO add in proper normal direction handling
                        diag1 = back_outboard - front_inboard
                        diag2 = back_inboard - front_outboard
                        cross = np.cross(diag1, diag2)
                        normal_direction = cross / np.linalg.norm(cross)

                        # Establish some sign conventions (positive z, then positive y, then positive x)
                        if normal_direction[2] < 0:
                            normal_direction = -normal_direction
                        elif normal_direction[2] == 0:
                            if normal_direction[1] < 0:
                                normal_direction = -normal_direction
                            elif normal_direction[1] == 0:
                                if normal_direction[0] < 0:
                                    normal_direction = -normal_direction

                        # Make the horseshoe vortex
                        inboard_vortex_point = (
                                0.75 * front_inboard +
                                0.25 * back_inboard
                        )
                        outboard_vortex_point = (
                                0.75 * front_outboard +
                                0.25 * back_inboard
                        )

                        vortex = HorseshoeVortex(
                            vertices=np.vstack((
                                inboard_vortex_point,
                                outboard_vortex_point
                            ))
                        )

                        new_panel = Panel(
                            vertices=vertices,
                            colocation_point=colocation_point,
                            normal_direction=normal_direction,
                            influencing_objects=[vortex]
                        )
                        self.panels.append(new_panel)

                        if wing.symmetric:
                            vortex_sym = HorseshoeVortex(
                                vertices=np.vstack((
                                    reflect_over_XZ_plane(outboard_vortex_point),
                                    reflect_over_XZ_plane(inboard_vortex_point)
                                ))
                            )

                            new_panel_sym = Panel(
                                vertices=reflect_over_XZ_plane(vertices),
                                colocation_point=reflect_over_XZ_plane(colocation_point),
                                normal_direction=reflect_over_XZ_plane(normal_direction),
                                influencing_objects=[vortex_sym]
                            )
                            self.panels.append(new_panel_sym)

        # # Calculate AIC matrix
        # ----------------------

        print("Calculating the influence matrix...")
        # Initialize AIC matrix
        n_panels = len(self.panels)
        AIC = np.zeros([n_panels, n_panels])

        # Naive approach to AIC calculation, takes a dummy long amount of time
        # AIC_naive = AIC
        # for colocation_num in range(n_panels):
        #     colocation_point = self.panels[colocation_num].colocation_point
        #     normal_direction = self.panels[colocation_num].normal_direction
        #     for vortex_num in range(n_panels):
        #         AIC_naive[colocation_num,vortex_num]=self.panels[vortex_num].influencing_objects[0].calculate_unit_influence(colocation_point)@normal_direction

        # Vectorized approach to AIC calculation, less human-readable but much, much faster
        # Make some vectorized variables
        c = np.zeros([n_panels, 3])  # Location of all colocation points
        n = np.zeros([n_panels, 3])  # Unit normals of all colocation points
        lv = np.zeros([n_panels, 3])  # Left vertex of all horseshoe vortices
        rv = np.zeros([n_panels, 3])  # Right vertex of all horseshoe vortices

        for panel_num in range(n_panels):
            c[panel_num, :] = self.panels[panel_num].colocation_point
            n[panel_num, :] = self.panels[panel_num].normal_direction
            lv[panel_num, :] = self.panels[panel_num].influencing_objects[0].vertices[0, :]
            rv[panel_num, :] = self.panels[panel_num].influencing_objects[0].vertices[1, :]

        Vij = self.calculate_Vij(c, lv, rv)

        # Calculate AIC matrix (Vij * normal vectors)
        n_tiled = np.tile(
            np.expand_dims(n, 1),
            [1, n_panels, 1]
        )
        AIC = np.sum(
            Vij * n_tiled,
            axis=2
        )

        # # # Calculate AIC Inverse
        # # -----------------------
        # print("Inverting the influence matrix...")
        # try:
        #     AIC_inv = np.linalg.inv(AIC)
        # except np.linalg.LinAlgError:
        #     raise Exception(
        #         "AIC matrix is singular and therefore cannot be inverted. Are you sure that you don't have multiple wings/panels stacked on top of each other? (Check the ""symmetry"" property of vertical stabilizers on the centerline, that's a common culprit.)")

        # # Calculate RHS
        # ---------------
        print("Calculating the freestream influence...")
        velocity = self.op_point.compute_freestream_velocity_geometry_axes()  # Direction the wind is GOING TO, in geometry axes coordinates

        local_velocity = np.tile(
            np.expand_dims(velocity, 0),
            [n_panels, 1]
        )  # this is a Nx3 vector representing the local velocity of the unperturbed flow. The first index refers to the colocation point number, the second refers to xyz.

        steady_influence = np.sum(local_velocity * n, axis=1)

        # TODO add rotation rates / control deflections to freestream_influence
        freestream_influence = steady_influence  # + blah blah blah

        # # Calculate Vortex Strengths
        # ----------------------------
        # Governing Equation: AIC @ Gamma + freestream_influence = 0
        print("LU factorizing the influence matrix...")
        lu, piv = sp_linalg.lu_factor(AIC)

        print("Calculating vortex strengths...")
        vortex_strengths = sp_linalg.lu_solve((lu, piv), -freestream_influence)

        # Perform assignments to panels
        for panel_num in range(len(self.panels)):
            panel = self.panels[panel_num]
            panel.influencing_objects[0].strength = vortex_strengths[panel_num]

        # # DEBUG only: Check your work
        # print("DEBUG ONLY: Checking work...")
        # check = AIC @ vortex_strengths + freestream_influence

        # # Calculate Near-Field Forces and Moments
        # -----------------------------------------
        print("Calculating near-field velocities...")
        vortex_centers = (lv + rv) / 2  # location of all vortex centers, where the near-field force is assumed to act
        vortex_centers_tiled = np.tile(
            np.expand_dims(
                vortex_centers,
                axis=1
            ),
            reps=[1, n_panels, 1]
        )  # First index is vortex center point #, second is vortex #, and third is xyz.

        # Basically from here until the "Vij = " line, we're just redoing the AIC calculation, but using vortex center points instead of colocation points
        Vij = self.calculate_Vij(vortex_centers, lv, rv)

        # Calculate Vi (local velocity at the ith vortex center point)
        Vi_x = Vij[:, :, 0] @ vortex_strengths + local_velocity[:, 0]
        Vi_y = Vij[:, :, 1] @ vortex_strengths + local_velocity[:, 1]
        Vi_z = Vij[:, :, 2] @ vortex_strengths + local_velocity[:, 2]
        Vi_x = np.expand_dims(Vi_x, axis=1)
        Vi_y = np.expand_dims(Vi_y, axis=1)
        Vi_z = np.expand_dims(Vi_z, axis=1)
        Vi = np.hstack((Vi_x, Vi_y, Vi_z))

        print("Calculating forces on each panel...")
        # Calculate li, the length of the bound segment of the horseshoe vortex filament
        li = rv - lv

        # Calculate Fi, the force on the ith panel. Note that this is in GEOMETRY AXES, not WIND AXES or BODY AXES.
        density = self.op_point.density
        li_norm = np.tile(np.expand_dims(np.linalg.norm(li, axis=1), axis=1), reps=[1, 3])
        Vi_cross_li_unit = np.cross(Vi, li / li_norm, axis=1)
        vortex_strengths_tiled = np.tile(np.expand_dims(vortex_strengths, axis=1), reps=[1, 3])

        Fi_geometry = density * Vi_cross_li_unit * vortex_strengths_tiled / 2  # TODO Figure out why the 2 is needed

        # Fi_geometry = (self.op_point.density *  # density
        #                np.cross(Vi, li, axis=1) *  # Vi cross li
        #                np.tile(
        #                    np.expand_dims(
        #                        vortex_strengths, axis=1
        #                    ),
        #                    reps=[1, 3]
        #                )
        #                )  # gamma

        # Perform assignments to panels
        for panel_num in range(len(self.panels)):
            panel = self.panels[panel_num]
            panel.force_geometry_axes = Fi_geometry[panel_num, :]

        # Calculate total forces
        print("Calculating total forces and moments...")

        Ftotal_geometry = np.sum(Fi_geometry, axis=0)  # Remember, this is in GEOMETRY AXES, not WIND AXES or BODY AXES.
        Faverage_geometry = np.mean(Fi_geometry, axis=0)
        print("Total aerodynamic forces (geometry axes): ", Ftotal_geometry)

        Ftotal_wind = np.transpose(self.op_point.compute_rotation_matrix_wind_to_geometry()) @ Ftotal_geometry
        print("Total aerodynamic forces (wind axes):", Ftotal_wind)

        # Calculate nondimensional forces
        CL = -Ftotal_wind[2] / (0.5 * self.op_point.density * self.op_point.velocity ** 2 * self.airplane.s_ref)
        CDi = -Ftotal_wind[0] / (0.5 * self.op_point.density * self.op_point.velocity ** 2 * self.airplane.s_ref)
        CY = Ftotal_wind[1] / (0.5 * self.op_point.density * self.op_point.velocity ** 2 * self.airplane.s_ref)
        print("CL: ", CL)
        print("CDi: ", CDi)
        print("CY: ", CY)
        print("CL/CDi: ", CL / CDi)

        # Calculate delta-cp
        for panel_num in range(len(self.panels)):
            panel = self.panels[panel_num]
            panel.force_normal = panel.force_geometry_axes @ panel.normal_direction
            panel.pressure_normal = panel.force_normal / panel.area()
            panel.delta_cp = panel.pressure_normal / (0.5 * self.op_point.density * self.op_point.velocity ** 2)

        print("VLM1 calculation complete!")

        # TODO make this object

    def calculate_Vij(self,
                      points,
                      left_vertices,
                      right_vertices,
                      ):
        # Calculates Vij, the velocity influence matrix (First index is colocation point number, second index is vortex number).
        # points: the list of points (Nx3) to calculate the velocity influence at.
        # left_vertices: list of left vertex points of the horseshoe vertices
        # right_vertices: list of right vertex points of the horseshoe vertices

        n_points = np.shape(points)[0]
        n_vortices = np.shape(left_vertices)[0]

        lv_tiled = np.tile(np.expand_dims(left_vertices, 0), [n_points, 1, 1])
        rv_tiled = np.tile(np.expand_dims(right_vertices, 0), [n_points, 1, 1])
        c_tiled = np.tile(np.expand_dims(points, 1), [1, n_vortices, 1])

        # Make a and b vectors.
        # a: Vector from all colocation points to all horseshoe vortex left  vertices, NxNx3. First index is colocation point #, second is vortex #, and third is xyz. N=n_panels
        # b: Vector from all colocation points to all horseshoe vortex right vertices, NxNx3. First index is colocation point #, second is vortex #, and third is xyz. N=n_panels
        # a[i,j,:] = lv[j,:] - c[i,:]
        # b[i,j,:] = rv[j,:] - c[i,:]
        a = c_tiled - lv_tiled
        b = c_tiled - rv_tiled
        x_hat = np.zeros([n_points, n_vortices, 3])
        x_hat[:, :, 0] = 1

        # Do some useful arithmetic
        a_cross_b = np.cross(a, b, axis=2)
        a_dot_b = np.sum(
            a * b,
            axis=2
        )
        a_cross_x = np.cross(a, x_hat, axis=2)
        a_dot_x = np.sum(
            a * x_hat,
            axis=2
        )
        b_cross_x = np.cross(b, x_hat, axis=2)
        b_dot_x = np.sum(
            b * x_hat,
            axis=2
        )
        norm_a = np.linalg.norm(a, axis=2)
        norm_b = np.linalg.norm(b, axis=2)
        norm_a_inv = 1 / norm_a
        norm_b_inv = 1 / norm_b

        # Check for the special case where the colocation point is along the bound vortex leg
        # Find where cross product is near zero, and set the dot product to infinity so that the value of the bound term is zero.
        singularity_indices = (
                np.abs(
                    np.linalg.norm(
                        a_cross_b, axis=2
                    )
                ) <= np.finfo(float).eps
        )
        a_dot_b[singularity_indices] = np.Inf

        # Calculate Vij
        term1 = (norm_a_inv + norm_b_inv) / (norm_a * norm_b + a_dot_b)
        term2 = (norm_a_inv) / (norm_a - a_dot_x)
        term3 = (norm_b_inv) / (norm_b - b_dot_x)
        term1 = np.tile(np.expand_dims(term1, 2), [1, 1, 3])
        term2 = np.tile(np.expand_dims(term2, 2), [1, 1, 3])
        term3 = np.tile(np.expand_dims(term3, 2), [1, 1, 3])

        Vij = 1 / (4 * np.pi) * (
                a_cross_b * term1 +
                a_cross_x * term2 -
                b_cross_x * term3
        )

        return Vij


    def draw_panels(self,
                    draw_colocation_points=False,
                    draw_panel_numbers=False,
                    draw_vortex_strengths=False,
                    draw_forces=False,
                    draw_pressures=False,
                    draw_pressures_as_vectors=False,
                    ):
        fig, ax = fig3d()
        n_panels = len(self.panels)

        if draw_vortex_strengths:
            # Calculate color bounds and box bounds
            min_strength = 0
            max_strength = 0
            for panel in self.panels:
                min_strength = min(min_strength, panel.influencing_objects[0].strength)
                max_strength = max(max_strength, panel.influencing_objects[0].strength)
            print("Colorbar min: ", min_strength)
            print("Colorbar max: ", max_strength)
        elif draw_pressures:
            min_delta_cp = 0
            max_delta_cp = 0
            for panel in self.panels:
                min_delta_cp = min(min_delta_cp, panel.delta_cp)
                max_delta_cp = max(max_delta_cp, panel.delta_cp)
            print("Colorbar min: ", min_delta_cp)
            print("Colorbar max: ", max_delta_cp)

        # Draw
        for panel_num in range(n_panels):
            sys.stdout.write('\r')
            sys.stdout.write("Drawing panel %i of %i" % (panel_num + 1, n_panels))
            sys.stdout.flush()
            panel = self.panels[panel_num]
            if draw_vortex_strengths:
                # Calculate colors and draw
                strength = panel.influencing_objects[0].strength
                normalized_strength = 1 * (strength - min_strength) / (max_strength - min_strength)
                colormap = mpl.cm.get_cmap('viridis')
                color = colormap(normalized_strength)
                panel.draw(
                    show=False,
                    fig_to_plot_on=fig,
                    ax_to_plot_on=ax,
                    draw_colocation_point=draw_colocation_points,
                    shading_color=color
                )
            elif draw_pressures:
                # Calculate colors and draw
                delta_cp = panel.delta_cp
                min_delta_cp = -8
                max_delta_cp = 2

                normalized_delta_cp = 1 * (delta_cp - min_delta_cp) / (max_delta_cp - min_delta_cp)
                colormap = mpl.cm.get_cmap('viridis')
                color = colormap(normalized_delta_cp)
                panel.draw(
                    show=False,
                    fig_to_plot_on=fig,
                    ax_to_plot_on=ax,
                    draw_colocation_point=draw_colocation_points,
                    shading_color=color
                )
            else:
                panel.draw(
                    show=False,
                    fig_to_plot_on=fig,
                    ax_to_plot_on=ax,
                    draw_colocation_point=draw_colocation_points,
                    shading_color=(0.5, 0.5, 0.5)
                )

            if draw_forces:
                force_scale = 10
                centroid = panel.centroid()

                tail = centroid
                head = centroid + force_scale * panel.force_geometry_axes

                x = np.array([tail[0], head[0]])
                y = np.array([tail[1], head[1]])
                z = np.array([tail[2], head[2]])

                ax.plot(x, y, z, color='#0A5E08')
            elif draw_pressures_as_vectors:
                pressure_scale = 10/10000
                centroid = panel.centroid()

                tail = centroid
                head = centroid + pressure_scale * panel.force_geometry_axes / panel.area()

                x = np.array([tail[0], head[0]])
                y = np.array([tail[1], head[1]])
                z = np.array([tail[2], head[2]])

                ax.plot(x, y, z, color='#0A5E08')


            if draw_panel_numbers:
                ax.text(
                    panel.colocation_point[0],
                    panel.colocation_point[1],
                    panel.colocation_point[2],
                    str(panel_num),
                )

        x, y, z, s = self.airplane.get_bounding_cube()
        ax.set_xlim3d((x - s, x + s))
        ax.set_ylim3d((y - s, y + s))
        ax.set_zlim3d((z - s, z + s))
        plt.tight_layout()
        plt.show()


class Panel:
    def __init__(self,
                 vertices=None,  # Nx3 np array, each row is a vector. Just used for drawing panel
                 colocation_point=None,  # 1x3 np array
                 normal_direction=None,  # 1x3 np array, nonzero
                 influencing_objects=[],  # List of Vortexes and Sources
                 ):
        self.vertices = np.array(vertices)
        self.colocation_point = np.array(colocation_point)
        self.normal_direction = np.array(normal_direction)
        self.influencing_objects = influencing_objects

        assert (np.shape(self.vertices)[0] >= 3)
        assert (np.shape(self.vertices)[1] == 3)

    def centroid(self):
        return np.mean(self.vertices, axis=0)

    def area(self):
        centroid = self.centroid()
        area = 0
        for i in range(len(self.vertices) - 1):
            area += 0.5 * np.linalg.norm(
                np.cross(
                    self.vertices[i, :] - centroid,
                    self.vertices[i + 1, :] - centroid
                )
            )
        return area

    def set_colocation_point_at_centroid(self):
        centroid = np.mean(self.vertices, axis=0)
        self.colocation_point = centroid

    def add_ring_vortex(self):
        pass

    def calculate_influence(self, point):
        influence = np.zeros([3])
        for object in self.influencing_objects:
            influence += object.calculate_unit_influence(point)
        return influence

    def draw(self,
             show=True,
             fig_to_plot_on=None,
             ax_to_plot_on=None,
             draw_colocation_point=True,
             shading_color=None,
             ):

        # Setup
        if fig_to_plot_on == None or ax_to_plot_on == None:
            fig, ax = fig3d()
            fig.set_size_inches(12, 9)
        else:
            fig = fig_to_plot_on
            ax = ax_to_plot_on

        # Plot vertices
        if not (self.vertices == None).all():
            quad = Poly3DCollection([self.vertices],
                                    cmap=mpl.cm.get_cmap('viridis'),
                                    )
            quad.set_edgecolor(None)
            quad.set_facecolor(shading_color)
            ax.add_collection3d(quad)

            # verts_to_draw = np.vstack((self.vertices, self.vertices[0, :]))
            # x = verts_to_draw[:, 0]
            # y = verts_to_draw[:, 1]
            # z = verts_to_draw[:, 2]
            # ax.plot(x, y, z, color='#be00cc', linestyle='-', linewidth=0.4)

        # Plot colocation point
        if draw_colocation_point and not (self.colocation_point == None).all():
            x = self.colocation_point[0]
            y = self.colocation_point[1]
            z = self.colocation_point[2]
            ax.scatter(x, y, z, color=(0, 0, 0), marker='.', s=0.5)

        if show:
            plt.show()


class HorseshoeVortex:
    # As coded, can only have two points not at infinity (3-leg horseshoe vortex)
    # Wake assumed to trail to infinity in the x-direction.
    def __init__(self,
                 vertices=None,  # 2x3 np array, left point first, then right.
                 strength=0,
                 ):
        self.vertices = np.array(vertices)
        self.strength = np.array(strength)

        assert (self.vertices.shape == (2, 3))

    def calculate_unit_influence(self, point):
        # Calculates the velocity induced at a point per unit vortex strength
        # Taken from Drela's Flight Vehicle Aerodynamics, pg. 132

        a = point - self.vertices[0, :]
        b = point - self.vertices[1, :]
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        x_hat = np.array([1, 0, 0])

        influence = 1 / (4 * np.pi) * (
                (np.cross(a, b) / (norm_a * norm_b + a @ b)) * (1 / norm_a + 1 / norm_b) +
                (np.cross(a, x_hat) / (norm_a - a @ x_hat)) / norm_a -
                (np.cross(b, x_hat) / (norm_b - b @ x_hat)) / norm_b
        )

        return influence


class Source:
    # A (3D) point source/sink.
    def __init__(self,
                 vertex=None,
                 strength=0,
                 ):
        self.vertex = np.array(vertex)
        self.strength = np.array(strength)

        assert (self.vertices.shape == (3))

    def calculate_unit_influence(self, point):
        r = self.vertices - point
        norm_r = np.linalg.norm(r)

        influence = 1 / (4 * np.pi * norm_r ** 2)

        return influence
