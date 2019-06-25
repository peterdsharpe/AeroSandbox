import autograd as np
from autograd import grad
import scipy.linalg as sp_linalg
from numba import jit
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from .Plotting import *
from .Geometry import *
from .Performance import *

import cProfile
import functools
import os


def profile(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        try:
            profiler.enable()
            ret = func(*args, **kwargs)
            profiler.disable()
            return ret
        finally:
            filename = os.path.expanduser(
                os.path.join('~', func.__name__ + '.pstat')
            )
            profiler.dump_stats(filename)
            profiler.print_stats()

    return wrapper


class AeroProblem:
    def __init__(self,
                 airplane=Airplane(),
                 op_point=OperatingPoint(),
                 ):
        self.airplane = airplane
        self.op_point = op_point


class vlm1(AeroProblem):
    # Traditional Vortex Lattice Method approach with quadrilateral paneling, horseshoe vortices from each one, etc.
    # Implemented exactly as The Good Book says (Drela, "Flight Vehicle Aerodynamics", p. 130-135)

    def run(self):
        print("Running VLM1 calculation...")
        self.make_panels()
        self.setup_geometry()
        self.setup_operating_point()
        self.calculate_vortex_strengths()
        self.calculate_forces()
        print("VLM1 calculation complete!")

    def make_panels(self):

        # # Make panels
        # -------------

        print("Making panels...")
        c = np.empty((0, 3))
        n = np.empty((0, 3))
        lv = np.empty((0, 3))
        rv = np.empty((0, 3))
        front_left_vertices = np.empty((0, 3))
        front_right_vertices = np.empty((0, 3))
        back_left_vertices = np.empty((0, 3))
        back_right_vertices = np.empty((0, 3))
        is_trailing_edge = np.empty((0), dtype=bool)
        for wing in self.airplane.wings:

            # Define number of chordwise points
            n_chordwise_coordinates = wing.chordwise_panels + 1

            # Get the chordwise coordinates
            if wing.chordwise_spacing == 'uniform':
                nondim_chordwise_coordinates = np.linspace(0, 1, n_chordwise_coordinates)
            elif wing.chordwise_spacing == 'cosine':
                nondim_chordwise_coordinates = 0.5 + 0.5 * np.cos(np.linspace(np.pi, 0, n_chordwise_coordinates))
            else:
                raise Exception("Bad value of wing.chordwise_spacing!")

            # Initialize an array of coordinates. Indices:
            #   Index 1: chordwise location
            #   Index 2: spanwise location
            #   Index 3: X, Y, or Z.
            wing_coordinates = np.empty((n_chordwise_coordinates, 0, 3))

            for section_num in range(len(wing.sections) - 1):

                # Define the relevant sections
                section = wing.sections[section_num]
                next_section = wing.sections[section_num + 1]

                # Define number of spanwise points
                n_spanwise_coordinates = section.spanwise_panels + 1

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

                section_coordinates = np.zeros(shape=(n_chordwise_coordinates, n_spanwise_coordinates, 3))

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

                        section_coordinates[chordwise_coordinate_num, spanwise_coordinate_num, :] = local_coordinate

                is_last_section = section_num == len(wing.sections) - 2
                if not is_last_section:
                    section_coordinates = section_coordinates[:, :-1, :]

                wing_coordinates = np.hstack((wing_coordinates, section_coordinates))

            front_inboard_vertices = wing_coordinates[:-1, :-1, :]
            front_outboard_vertices = wing_coordinates[:-1, 1:, :]
            back_inboard_vertices = wing_coordinates[1:, :-1, :]
            back_outboard_vertices = wing_coordinates[1:, 1:, :]
            is_trailing_edge_this_wing = np.vstack((
                np.zeros((wing_coordinates.shape[0] - 2, wing_coordinates.shape[1] - 1), dtype=bool),
                np.ones((1, wing_coordinates.shape[1] - 1), dtype=bool)
            ))

            colocation_points = (
                    0.25 * (front_inboard_vertices + front_outboard_vertices) / 2 +
                    0.75 * (back_inboard_vertices + back_outboard_vertices) / 2
            )

            diag1 = back_outboard_vertices - front_inboard_vertices
            diag2 = back_inboard_vertices - front_outboard_vertices
            cross = np.cross(diag1, diag2, axis=2)
            normal_directions = cross / np.expand_dims(np.linalg.norm(cross, axis=2),
                                                       axis=2)  # TODO add in proper normal direction handling

            # Make the horseshoe vortex
            inboard_vortex_points = (
                    0.75 * front_inboard_vertices +
                    0.25 * back_inboard_vertices
            )
            outboard_vortex_points = (
                    0.75 * front_outboard_vertices +
                    0.25 * back_outboard_vertices
            )

            colocation_points = np.reshape(colocation_points, (-1, 3), order='F')
            normal_directions = np.reshape(normal_directions, (-1, 3), order='F')
            inboard_vortex_points = np.reshape(inboard_vortex_points, (-1, 3), order='F')
            outboard_vortex_points = np.reshape(outboard_vortex_points, (-1, 3), order='F')
            front_inboard_vertices = np.reshape(front_inboard_vertices, (-1, 3), order='F')
            front_outboard_vertices = np.reshape(front_outboard_vertices, (-1, 3), order='F')
            back_inboard_vertices = np.reshape(back_inboard_vertices, (-1, 3), order='F')
            back_outboard_vertices = np.reshape(back_outboard_vertices, (-1, 3), order='F')
            is_trailing_edge_this_wing = np.reshape(is_trailing_edge_this_wing, (-1), order='F')

            c = np.vstack((c, colocation_points))
            n = np.vstack((n, normal_directions))
            lv = np.vstack((lv, inboard_vortex_points))
            rv = np.vstack((rv, outboard_vortex_points))
            front_left_vertices = np.vstack((front_left_vertices, front_inboard_vertices))
            front_right_vertices = np.vstack((front_right_vertices, front_outboard_vertices))
            back_left_vertices = np.vstack((back_left_vertices, back_inboard_vertices))
            back_right_vertices = np.vstack((back_right_vertices, back_outboard_vertices))
            is_trailing_edge = np.hstack((is_trailing_edge, is_trailing_edge_this_wing))

            if wing.symmetric:
                reflect_over_XZ_plane(inboard_vortex_points)
                reflect_over_XZ_plane(outboard_vortex_points)
                reflect_over_XZ_plane(colocation_points)
                reflect_over_XZ_plane(normal_directions)
                reflect_over_XZ_plane(front_inboard_vertices)
                reflect_over_XZ_plane(front_outboard_vertices)
                reflect_over_XZ_plane(back_inboard_vertices)
                reflect_over_XZ_plane(back_outboard_vertices)

                c = np.vstack((c, colocation_points))
                n = np.vstack((n, normal_directions))
                lv = np.vstack((lv, outboard_vortex_points))
                rv = np.vstack((rv, inboard_vortex_points))
                front_left_vertices = np.vstack((front_left_vertices, front_outboard_vertices))
                front_right_vertices = np.vstack((front_right_vertices, front_inboard_vertices))
                back_left_vertices = np.vstack((back_left_vertices, back_outboard_vertices))
                back_right_vertices = np.vstack((back_right_vertices, back_inboard_vertices))
                is_trailing_edge = np.hstack((is_trailing_edge, is_trailing_edge_this_wing))

        # Put normals in the "right" direction
        # Algorithm: First, try to make z positive. Then y, then x.
        n[n[:, 0] < 0] *= -1
        n[n[:, 1] < 0] *= -1
        n[n[:, 2] < 0] *= -1

        self.c = c
        self.n = n
        self.lv = lv
        self.rv = rv
        self.front_left_vertices = front_left_vertices
        self.front_right_vertices = front_right_vertices
        self.back_left_vertices = back_left_vertices
        self.back_right_vertices = back_right_vertices
        self.n_panels = len(c)
        self.is_trailing_edge = is_trailing_edge

    def setup_geometry(self):
        # # Calculate AIC matrix
        # ----------------------
        print("Calculating the colocation influence matrix...")

        # Python Mode
        self.Vij_colocations = self.calculate_Vij(self.c)
        # Numba Mode
        # self.Vij_colocations = self.calculate_Vij_jit(self.c, self.lv, self.rv)

        # Vij_colocations: [points, vortices, xyz]
        # n: [points, xyz]
        n_expanded = np.expand_dims(self.n, 1)

        # AIC = (Vij * normal vectors)
        self.AIC = np.sum(
            self.Vij_colocations * n_expanded,
            axis=2
        )

        # # Calculate Vij at vortex centers for force calculation
        # -------------------------------------------------------
        print("Calculating the vortex center influence matrix...")
        self.vortex_centers = (
                                      self.lv + self.rv) / 2  # location of all vortex centers, where the near-field force is assumed to act

        # Redoing the AIC calculation, but using vortex center points instead of colocation points
        # Python Mode
        self.Vij_centers = self.calculate_Vij(self.vortex_centers)
        # Numba Mode
        # self.Vij_centers = self.calculate_Vij_jit(self.vortex_centers, self.lv, self.rv)

        # # LU Decomposition on AIC
        # -------------------------
        print("LU factorizing the AIC matrix...")
        self.lu, self.piv = sp_linalg.lu_factor(self.AIC)

    def setup_operating_point(self):

        print("Calculating the freestream influence...")
        self.steady_freestream_velocity = self.op_point.compute_freestream_velocity_geometry_axes() * np.ones(
            (self.n_panels, 1))  # Direction the wind is GOING TO, in geometry axes coordinates
        self.rotation_freestream_velocities = np.zeros(
            (self.n_panels, 3))  # TODO Make this actually be the rotational velocity

        self.freestream_velocities = self.steady_freestream_velocity + self.rotation_freestream_velocities  # Nx3, represents the freestream velocity at each panel colocation point (c)

        self.freestream_influences = np.sum(self.freestream_velocities * self.n, axis=1)

    def calculate_vortex_strengths(self):
        # # Calculate Vortex Strengths
        # ----------------------------
        # Governing Equation: AIC @ Gamma + freestream_influence = 0
        print("Calculating vortex strengths...")
        self.vortex_strengths = sp_linalg.lu_solve((self.lu, self.piv), -self.freestream_influences)

    def calculate_forces(self):
        # # Calculate Near-Field Forces and Moments
        # -----------------------------------------
        # Governing Equation: The force on a straight, small vortex filament is F = rho * V x l * gamma, where rho is density, V is the velocity vector, x is the cross product operator, l is the vector of the filament itself, and gamma is the circulation.

        print("Calculating forces on each panel...")
        # Calculate Vi (local velocity at the ith vortex center point)
        Vi_x = self.Vij_centers[:, :, 0] @ self.vortex_strengths + self.freestream_velocities[:, 0]
        Vi_y = self.Vij_centers[:, :, 1] @ self.vortex_strengths + self.freestream_velocities[:, 1]
        Vi_z = self.Vij_centers[:, :, 2] @ self.vortex_strengths + self.freestream_velocities[:, 2]
        Vi_x = np.expand_dims(Vi_x, axis=1)
        Vi_y = np.expand_dims(Vi_y, axis=1)
        Vi_z = np.expand_dims(Vi_z, axis=1)
        Vi = np.hstack((Vi_x, Vi_y, Vi_z))

        # Calculate li, the length of the bound segment of the horseshoe vortex filament
        self.li = self.rv - self.lv

        # Calculate Fi_geometry, the force on the ith panel. Note that this is in GEOMETRY AXES, not WIND AXES or BODY AXES.
        density = self.op_point.density
        Vi_cross_li = np.cross(Vi, self.li, axis=1)
        vortex_strengths_expanded = np.expand_dims(self.vortex_strengths, axis=1)
        self.Fi_geometry = density * Vi_cross_li * vortex_strengths_expanded

        # Calculate total forces
        print("Calculating total forces and moments...")
        self.Ftotal_geometry = np.sum(self.Fi_geometry,
                                      axis=0)  # Remember, this is in GEOMETRY AXES, not WIND AXES or BODY AXES.
        print("Total aerodynamic forces (geometry axes): ", self.Ftotal_geometry)

        self.Ftotal_wind = np.transpose(self.op_point.compute_rotation_matrix_wind_to_geometry()) @ self.Ftotal_geometry
        print("Total aerodynamic forces (wind axes):", self.Ftotal_wind)

        # Calculate nondimensional forces
        qS = self.op_point.dynamic_pressure() * self.airplane.s_ref
        self.CL = -self.Ftotal_wind[2] / qS
        self.CDi = -self.Ftotal_wind[0] / qS
        self.CY = self.Ftotal_wind[1] / qS
        print("CL: ", self.CL)
        print("CDi: ", self.CDi)
        print("CY: ", self.CY)
        print("CL/CDi: ", self.CL / self.CDi)

    def calculate_delta_cp(self):
        # Find the area of each panel ()
        front_to_back = 0.5 * (
                self.front_left_vertices + self.front_right_vertices - self.back_left_vertices - self.back_right_vertices)
        self.areas_approx = np.linalg.norm(self.li, axis=1) * np.linalg.norm(front_to_back, axis=1)

        # Calculate panel data
        self.Fi_normal = np.einsum('ij,ij->i', self.Fi_geometry, self.n)
        self.pressure_normal = self.Fi_normal / self.areas_approx
        self.delta_cp = self.pressure_normal / self.op_point.dynamic_pressure()

    def get_induced_velocity_at_point(self, point):
        # Input: a Nx3 numpy array of points that you would like to know the induced velocities at.
        # Output: a Nx3 numpy array of the induced velocities at those points.
        point = np.reshape(point, (-1, 3))

        Vij = self.calculate_Vij(point)

        vortex_strengths_expanded = np.expand_dims(self.vortex_strengths, 1)

        # freestream = self.op_point.compute_freestream_velocity_geometry_axes()
        # V_x = Vij[:, :, 0] @ vortex_strengths_expanded + freestream[0]
        # V_y = Vij[:, :, 1] @ vortex_strengths_expanded + freestream[1]
        # V_z = Vij[:, :, 2] @ vortex_strengths_expanded + freestream[2]

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

    def calculate_Vij(self, points):
        # Calculates Vij, the velocity influence matrix (First index is colocation point number, second index is vortex number).
        # points: the list of points (Nx3) to calculate the velocity influence at.
        points = np.reshape(points, (-1, 3))

        n_points = len(points)
        n_vortices = len(self.lv)

        c_tiled = np.expand_dims(points, 1)

        # Make a and b vectors.
        # a: Vector from all colocation points to all horseshoe vortex left  vertices, NxNx3. First index is colocation point #, second is vortex #, and third is xyz. N=n_panels
        # b: Vector from all colocation points to all horseshoe vortex right vertices, NxNx3. First index is colocation point #, second is vortex #, and third is xyz. N=n_panels
        # a[i,j,:] = c[i,:] - lv[j,:]
        # b[i,j,:] = c[i,:] - rv[j,:]
        a = c_tiled - self.lv
        b = c_tiled - self.rv
        # x_hat = np.zeros([n_points, n_vortices, 3])
        # x_hat[:, :, 0] = 1

        # Do some useful arithmetic
        a_cross_b = np.cross(a, b, axis=2)
        #     np.dstack((
        #     ay * bz - az * by,
        #     az * bx - ax * bz,
        #     ax * by - ay * bx
        # ))  # np.cross(a, b, axis=2)

        a_dot_b = np.einsum('ijk,ijk->ij', a, b)  # np.sum(a * b, axis=2)

        a_cross_x = np.zeros((n_points, n_vortices, 3))
        a_cross_x[:, :, 1] = a[:, :, 2]
        a_cross_x[:, :, 2] = -a[:, :, 1]
        # a_cross_x = np.dstack((
        #     np.zeros((n_points, n_vortices)),
        #     a[:,:,2],
        #     -a[:,:,1]
        # ))  # np.cross(a, x_hat, axis=2)

        a_dot_x = a[:, :, 0]  # np.sum(a * x_hat,axis=2)

        b_cross_x = np.zeros((n_points, n_vortices, 3))
        b_cross_x[:, :, 1] = b[:, :, 2]
        b_cross_x[:, :, 2] = -b[:, :, 1]
        # b_cross_x = np.dstack((
        #     np.zeros((n_points, n_vortices)),
        #     b[:,:,2],
        #     -b[:,:,1]
        # ))  # np.cross(b, x_hat, axis=2)

        b_dot_x = b[:, :, 0]  # np.sum(b * x_hat,axis=2)

        norm_a = np.linalg.norm(a, axis=2)
        # np.power(
        #     np.sum(
        #         a * a, axis=2
        #     ),
        #     0.5
        # )  #
        norm_b = np.linalg.norm(b, axis=2)
        # np.power(
        #     np.sum(
        #         b * b, axis=2
        #     ),
        #     0.5
        # )  #
        norm_a_inv = 1 / norm_a
        norm_b_inv = 1 / norm_b

        # Check for the special case where the colocation point is along the bound vortex leg
        # Find where cross product is near zero, and set the dot product to infinity so that the value of the bound term is zero.
        bound_vortex_singularity_indices = (
                np.einsum('ijk,ijk->ij', a_cross_b, a_cross_b)  # norm(a_cross_b) ** 2
                < 3.0e-16)
        a_dot_b[bound_vortex_singularity_indices] = np.inf  # something non-infinitesimal
        left_vortex_singularity_indices = (
                np.einsum('ijk,ijk->ij', a_cross_x, a_cross_x)
                < 3.0e-16
        )
        a_dot_x[left_vortex_singularity_indices] = np.inf
        right_vortex_singularity_indices = (
                np.einsum('ijk,ijk->ij', b_cross_x, b_cross_x)
                < 3.0e-16
        )
        b_dot_x[right_vortex_singularity_indices] = np.inf

        # Calculate Vij
        term1 = (norm_a_inv + norm_b_inv) / (norm_a * norm_b + a_dot_b)
        term2 = (norm_a_inv) / (norm_a - a_dot_x)
        term3 = (norm_b_inv) / (norm_b - b_dot_x)
        term1 = np.expand_dims(term1, 2)
        term2 = np.expand_dims(term2, 2)
        term3 = np.expand_dims(term3, 2)

        Vij = 1 / (4 * np.pi) * (
                a_cross_b * term1 +
                a_cross_x * term2 -
                b_cross_x * term3
        )

        return Vij

    @staticmethod
    @jit()
    def calculate_Vij_jit(points, lv, rv):
        # Calculates Vij, the velocity influence matrix (First index is colocation point number, second index is vortex number).
        # points: the list of points (Nx3) to calculate the velocity influence at.
        points = np.reshape(points, (-1, 3))

        n_points = len(points)
        n_vortices = len(lv)

        c_tiled = np.expand_dims(points, 1)

        # Make a and b vectors.
        # a: Vector from all colocation points to all horseshoe vortex left  vertices, NxNx3. First index is colocation point #, second is vortex #, and third is xyz. N=n_panels
        # b: Vector from all colocation points to all horseshoe vortex right vertices, NxNx3. First index is colocation point #, second is vortex #, and third is xyz. N=n_panels
        # a[i,j,:] = c[i,:] - lv[j,:]
        # b[i,j,:] = c[i,:] - rv[j,:]
        a = c_tiled - lv
        b = c_tiled - rv
        # x_hat = np.zeros([n_points, n_vortices, 3])
        # x_hat[:, :, 0] = 1

        ax = a[:, :, 0]
        ay = a[:, :, 1]
        az = a[:, :, 2]
        bx = b[:, :, 0]
        by = b[:, :, 1]
        bz = b[:, :, 2]

        # Do some useful arithmetic
        a_cross_b = np.dstack((
            ay * bz - az * by,
            az * bx - ax * bz,
            ax * by - ay * bx
        ))  # np.cross(a, b, axis=2)
        a_dot_b = np.sum(
            a * b,
            axis=2
        )
        a_cross_x = np.dstack((
            np.zeros((n_points, n_vortices)),
            az,
            -ay
        ))  # np.cross(a, x_hat, axis=2)
        a_dot_x = ax  # np.sum(a * x_hat,axis=2)

        b_cross_x = np.dstack((
            np.zeros((n_points, n_vortices)),
            bz,
            -by
        ))  # np.cross(b, x_hat, axis=2)
        b_dot_x = bx  # np.sum(b * x_hat,axis=2)

        norm_a = np.power(
            np.sum(
                a * a, axis=2
            ),
            0.5
        )  # np.linalg.norm(a, axis=2)
        norm_b = np.power(
            np.sum(
                b * b, axis=2
            ),
            0.5
        )  # np.linalg.norm(b, axis=2)
        norm_a_inv = 1 / norm_a
        norm_b_inv = 1 / norm_b

        # Check for the special case where the colocation point is along the bound vortex leg
        # Find where cross product is near zero, and set the dot product to infinity so that the value of the bound term is zero.
        singularity_indices = (
                np.sum(a_cross_b * a_cross_b, axis=2)
                < 3.0e-16  # Approximately eps
            # np.abs(
            #     np.linalg.norm(a_cross_b, axis=2)
            # ) <= np.finfo(float).eps
        )
        a_dot_b = a_dot_b * singularity_indices.astype(int)  # something non-infinitesimal

        # Calculate Vij
        term1 = (norm_a_inv + norm_b_inv) / (norm_a * norm_b + a_dot_b)
        term2 = (norm_a_inv) / (norm_a - a_dot_x)
        term3 = (norm_b_inv) / (norm_b - b_dot_x)
        term1 = np.expand_dims(term1, 2)
        term2 = np.expand_dims(term2, 2)
        term3 = np.expand_dims(term3, 2)

        Vij = 1 / (4 * np.pi) * (
                a_cross_b * term1 +
                a_cross_x * term2 -
                b_cross_x * term3
        )

        return Vij

    def calculate_streamlines(self):
        # Calculates streamlines eminating from the trailing edges of all surfaces.
        # "streamlines" is a MxNx3 array, where M is the index of the streamline number, N is the index of the timestep, and the last index is xyz

        # Constants
        n_steps = 100  # minimum of 2
        length = 1  # meter

        # Resolution
        length_per_step = length / n_steps
        # dt = length / self.op_point.velocity / n_steps

        # Seed points
        seed_points = (0.5 * (self.back_left_vertices + self.back_right_vertices))[self.is_trailing_edge]
        n_streamlines = len(seed_points)

        # Initialize
        streamlines = np.zeros((n_streamlines, n_steps, 3))
        streamlines[:, 0, :] = seed_points

        # Iterate
        for step_num in range(1, n_steps):
            update_amount = self.get_velocity_at_point(streamlines[:, step_num - 1, :])
            update_amount = update_amount / np.expand_dims(np.linalg.norm(update_amount, axis = 1), axis = 1)
            update_amount *= length_per_step
            streamlines[:, step_num, :] = streamlines[:, step_num - 1, :] + update_amount


        self.streamlines = streamlines

    def draw(self,
             draw_delta_cp=True,
             draw_streamlines=True,
             ):

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

        # Initialize Plotter
        plotter = pv.Plotter()

        if draw_delta_cp:
            if not hasattr(self, 'delta_cp'):
                self.calculate_delta_cp()

            scalars = np.minimum(np.maximum(self.delta_cp, -1), 1)
            cmap = plt.cm.get_cmap('viridis')
            plotter.add_mesh(wing_surfaces, scalars=scalars, cmap=cmap, color='tan', show_edges=True,
                             smooth_shading=True)
            plotter.add_scalar_bar(title="Pressure Coefficient", n_labels=5, shadow=True, font_family='arial')

        if draw_streamlines:
            if not hasattr(self, 'streamlines'):
                self.calculate_streamlines()

            for streamline_num in range(len(self.streamlines)):
                plotter.add_lines(self.streamlines[streamline_num, :, :], width=1.5, color='#50C7C7')

        # Do the plotting
        plotter.show_grid(color='#444444')
        plotter.set_background(color="black")
        plotter.show(cpos=(-1, -1, 1), full_screen=False)

    def draw_legacy(self,
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
                panel.draw_legacy(
                    show=False,
                    fig_to_plot_on=fig,
                    ax_to_plot_on=ax,
                    draw_colocation_point=draw_colocation_points,
                    shading_color=color
                )
            elif draw_pressures:
                # Calculate colors and draw
                delta_cp = panel.delta_cp
                min_delta_cp = -2
                max_delta_cp = 2

                normalized_delta_cp = 1 * (delta_cp - min_delta_cp) / (max_delta_cp - min_delta_cp)
                colormap = mpl.cm.get_cmap('viridis')
                color = colormap(normalized_delta_cp)
                panel.draw_legacy(
                    show=False,
                    fig_to_plot_on=fig,
                    ax_to_plot_on=ax,
                    draw_colocation_point=draw_colocation_points,
                    shading_color=color
                )
            else:
                panel.draw_legacy(
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
                pressure_scale = 10 / 10000
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

#
# class Panel:
#     def __init__(self,
#                  vertices=None,  # Nx3 np array, each row is a vector. Just used for drawing panel
#                  colocation_point=None,  # 1x3 np array
#                  normal_direction=None,  # 1x3 np array, nonzero
#                  influencing_objects=[],  # List of Vortexes and Sources
#                  ):
#         self.vertices = np.array(vertices)
#         self.colocation_point = np.array(colocation_point)
#         self.normal_direction = np.array(normal_direction)
#         self.influencing_objects = influencing_objects
#
#         assert (np.shape(self.vertices)[0] >= 3)
#         assert (np.shape(self.vertices)[1] == 3)
#
#     def centroid(self):
#         return np.mean(self.vertices, axis=0)
#
#     def area(self):
#         centroid = self.centroid()
#         area = 0
#         for i in range(len(self.vertices) - 1):
#             area += 0.5 * np.linalg.norm(
#                 np.cross(
#                     self.vertices[i, :] - centroid,
#                     self.vertices[i + 1, :] - centroid
#                 )
#             )
#         return area
#
#     def set_colocation_point_at_centroid(self):
#         centroid = np.mean(self.vertices, axis=0)
#         self.colocation_point = centroid
#
#     def add_ring_vortex(self):
#         pass
#
#     def calculate_influence(self, point):
#         influence = np.zeros([3])
#         for object in self.influencing_objects:
#             influence += object.calculate_unit_influence(point)
#         return influence
#
#     def draw(self,
#              show=True,
#              fig_to_plot_on=None,
#              ax_to_plot_on=None,
#              draw_colocation_point=True,
#              shading_color=None,
#              ):
#
#         # Setup
#         if fig_to_plot_on == None or ax_to_plot_on == None:
#             fig, ax = fig3d()
#             fig.set_size_inches(12, 9)
#         else:
#             fig = fig_to_plot_on
#             ax = ax_to_plot_on
#
#         # Plot vertices
#         if not (self.vertices == None).all():
#             quad = Poly3DCollection([self.vertices],
#                                     cmap=mpl.cm.get_cmap('viridis'),
#                                     )
#             quad.set_edgecolor(None)
#             quad.set_facecolor(shading_color)
#             ax.add_collection3d(quad)
#
#             # verts_to_draw = np.vstack((self.vertices, self.vertices[0, :]))
#             # x = verts_to_draw[:, 0]
#             # y = verts_to_draw[:, 1]
#             # z = verts_to_draw[:, 2]
#             # ax.plot(x, y, z, color='#be00cc', linestyle='-', linewidth=0.4)
#
#         # Plot colocation point
#         if draw_colocation_point and not (self.colocation_point == None).all():
#             x = self.colocation_point[0]
#             y = self.colocation_point[1]
#             z = self.colocation_point[2]
#             ax.scatter(x, y, z, color=(0, 0, 0), marker='.', s=0.5)
#
#         if show:
#             plt.show()
#
#
# class HorseshoeVortex:
#     # As coded, can only have two points not at infinity (3-leg horseshoe vortex)
#     # Wake assumed to trail to infinity in the x-direction.
#     def __init__(self,
#                  vertices=None,  # 2x3 np array, left point first, then right.
#                  strength=0,
#                  ):
#         self.vertices = np.array(vertices)
#         self.strength = np.array(strength)
#
#         assert (self.vertices.shape == (2, 3))
#
#     def calculate_unit_influence(self, point):
#         # Calculates the velocity induced at a point per unit vortex strength
#         # Taken from Drela's Flight Vehicle Aerodynamics, pg. 132
#
#         a = point - self.vertices[0, :]
#         b = point - self.vertices[1, :]
#         norm_a = np.linalg.norm(a)
#         norm_b = np.linalg.norm(b)
#         x_hat = np.array([1, 0, 0])
#
#         influence = 1 / (4 * np.pi) * (
#                 (np.cross(a, b) / (norm_a * norm_b + a @ b)) * (1 / norm_a + 1 / norm_b) +
#                 (np.cross(a, x_hat) / (norm_a - a @ x_hat)) / norm_a -
#                 (np.cross(b, x_hat) / (norm_b - b @ x_hat)) / norm_b
#         )
#
#         return influence
#
#
# class Source:
#     # A (3D) point source/sink.
#     def __init__(self,
#                  vertex=None,
#                  strength=0,
#                  ):
#         self.vertex = np.array(vertex)
#         self.strength = np.array(strength)
#
#         assert (self.vertices.shape == (3))
#
#     def calculate_unit_influence(self, point):
#         r = self.vertices - point
#         norm_r = np.linalg.norm(r)
#
#         influence = 1 / (4 * np.pi * norm_r ** 2)
#
#         return influence
