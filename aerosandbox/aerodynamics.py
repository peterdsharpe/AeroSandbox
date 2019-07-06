import autograd.numpy as np
from autograd import grad
import scipy.linalg as sp_linalg
from numba import jit
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from .plotting import *
from .geometry import *
from .performance import *

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
            # profiler.print_stats()

    return wrapper


class AeroProblem:
    def __init__(self,
                 airplane, # Object of Airplane class
                 op_point, # Object of OperatingPoint class
                 ):
        self.airplane = airplane
        self.op_point = op_point


class vlm1(AeroProblem):
    # NOTE: USE VLM2 INSTEAD OF THIS; VLM1 HAS BEEN COMPLETELY SUPERSEDED IN PERFORMANCE AND FUNCTIONALITY BY VLM2.
    # Traditional vortex-lattice-method approach with quadrilateral paneling, horseshoe vortices from each one, etc.
    # Implemented exactly as The Good Book says (Drela, "Flight Vehicle Aerodynamics", p. 130-135)

    @profile
    def run(self, verbose=True):
        self.verbose = verbose

        if self.verbose: print("Running VLM1 calculation...")
        # Deprecation warning (use VLM2 instead)
        if self.verbose: print("WARNING! VLM1 has been wholly eclipsed in performance and functionality by VLM2. The VLM1 source code has been left intact for validation purposes and backwards-compatibility, but it will not be supported going forward.")


        self.make_panels()
        self.setup_geometry()
        self.setup_operating_point()
        self.calculate_vortex_strengths()
        self.calculate_forces()

        if self.verbose: print("VLM1 calculation complete!")

    def make_panels(self):

        # # Make panels
        # -------------

        if self.verbose: print("Making panels...")
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
            n_chordwise_coordinates = wing.vlm_chordwise_panels + 1

            # Get the chordwise coordinates
            if wing.vlm_chordwise_spacing == 'uniform':
                nondim_chordwise_coordinates = np.linspace(0, 1, n_chordwise_coordinates)
            elif wing.vlm_chordwise_spacing == 'cosine':
                nondim_chordwise_coordinates = cosspace(n_points=n_chordwise_coordinates)
            else:
                raise Exception("Bad value of wing.vlm_chordwise_spacing!")

            # Initialize an array of coordinates. Indices:
            #   Index 1: chordwise location
            #   Index 2: spanwise location
            #   Index 3: X, Y, or Z.
            wing_coordinates = np.empty((n_chordwise_coordinates, 0, 3))

            # Initialize an array of normal vectors. Indices:
            #   Index 1: chordwise location
            #   Index 2: spanwise location
            #   Index 3: X, Y, or Z.
            wing_normals = np.empty((wing.vlm_chordwise_panels, 0, 3))

            for XSec_number in range(len(wing.xsecs) - 1):

                # Define the relevant cross sections (xsecs)
                xsec = wing.xsecs[XSec_number]
                next_xsec = wing.xsecs[XSec_number + 1]

                # Define number of spanwise points
                n_spanwise_coordinates = xsec.vlm_spanwise_panels + 1

                # Get the spanwise coordinates
                if xsec.vlm_spanwise_spacing == 'uniform':
                    nondim_spanwise_coordinates = np.linspace(0, 1, n_spanwise_coordinates)
                elif xsec.vlm_spanwise_spacing == 'cosine':
                    nondim_spanwise_coordinates = cosspace(n_points=n_spanwise_coordinates)
                else:
                    raise Exception("Bad value of section.vlm_spanwise_spacing!")

                # Get the corners of the WingXSec
                xsec_xyz_le = xsec.xyz_le + wing.xyz_le
                xsec_xyz_te = xsec.xyz_te() + wing.xyz_le
                next_xsec_xyz_le = next_xsec.xyz_le + wing.xyz_le
                next_xsec_xyz_te = next_xsec.xyz_te() + wing.xyz_le

                section_coordinates = np.empty(shape=(n_chordwise_coordinates, n_spanwise_coordinates, 3))

                # Dimensionalize the chordwise and spanwise coordinates using the corners
                for spanwise_coordinate_num in range(len(nondim_spanwise_coordinates)):
                    nondim_spanwise_coordinate = nondim_spanwise_coordinates[spanwise_coordinate_num]

                    local_xyz_le = ((1 - nondim_spanwise_coordinate) * xsec_xyz_le +
                                    (nondim_spanwise_coordinate) * next_xsec_xyz_le)
                    local_xyz_te = ((1 - nondim_spanwise_coordinate) * xsec_xyz_te +
                                    (nondim_spanwise_coordinate) * next_xsec_xyz_te)

                    for chordwise_coordinate_num in range(len(nondim_chordwise_coordinates)):
                        nondim_chordwise_coordinate = nondim_chordwise_coordinates[chordwise_coordinate_num]

                        local_coordinate = ((1 - nondim_chordwise_coordinate) * local_xyz_le +
                                            (nondim_chordwise_coordinate) * local_xyz_te)

                        section_coordinates[chordwise_coordinate_num, spanwise_coordinate_num, :] = local_coordinate

                is_last_section = XSec_number == len(wing.xsecs) - 2
                if not is_last_section:
                    section_coordinates = section_coordinates[:, :-1, :]

                wing_coordinates = np.concatenate((wing_coordinates, section_coordinates), axis=1)

                # Calculate the WingXSec's normal directions
                xsec_chord_vector = xsec_xyz_te - xsec_xyz_le  # vector of the chordline of this section
                next_xsec_chord_vector = next_xsec_xyz_te - next_xsec_xyz_le  # vector of the chordline of the next section
                quarter_chord_vector = (
                        (0.75 * next_xsec_xyz_le + 0.25 * next_xsec_xyz_te) -
                        (0.75 * xsec_xyz_le + 0.25 * xsec_xyz_te)
                )  # vector from the quarter-chord of the current section to the quarter-chord of the next section

                xsec_up = np.cross(xsec_chord_vector, quarter_chord_vector)
                xsec_up /= np.linalg.norm(xsec_up)
                xsec_back = xsec_chord_vector / np.linalg.norm(xsec_chord_vector)

                next_xsec_up = np.cross(next_xsec_chord_vector, quarter_chord_vector)
                next_xsec_up /= np.linalg.norm(next_xsec_up)
                next_xsec_back = next_xsec_chord_vector / np.linalg.norm(next_xsec_chord_vector)

                nondim_chordwise_colocation_coordinates = 0.25 * nondim_chordwise_coordinates[
                                                                 :-1] + 0.75 * nondim_chordwise_coordinates[1:]

                xsec_normals_2d = xsec.airfoil.get_normal_direction_at_chord_fraction(
                    nondim_chordwise_colocation_coordinates)  # Nx2 array of normal directions
                next_xsec_normals_2d = next_xsec.airfoil.get_normal_direction_at_chord_fraction(
                    nondim_chordwise_colocation_coordinates)

                xsec_normals = (
                        xsec_up * np.expand_dims(xsec_normals_2d[:, 1], axis=1) +
                        xsec_back * np.expand_dims(xsec_normals_2d[:, 0], axis=1)
                )
                next_xsec_normals = (
                        next_xsec_up * np.expand_dims(next_xsec_normals_2d[:, 1], axis=1) +
                        next_xsec_back * np.expand_dims(next_xsec_normals_2d[:, 0], axis=1)
                )

                nondim_spanwise_colocation_coordinates = 0.5 * nondim_spanwise_coordinates[
                                                               :-1] + 0.5 * nondim_spanwise_coordinates[1:]

                # Index 0: chordwise_coordinates
                # Index 1: spanwise_coordinates
                # Index 2: xyz
                section_normals = (
                        np.expand_dims(xsec_normals, axis=1) * (
                        1 - np.reshape(nondim_spanwise_colocation_coordinates, (1, -1, 1)))
                        + np.expand_dims(next_xsec_normals, axis=1) * np.reshape(nondim_spanwise_colocation_coordinates,
                                                                                 (1, -1, 1))
                )

                wing_normals = np.concatenate((wing_normals, section_normals), axis=1)

                # Get the corners of each panel
            front_inboard_vertices = wing_coordinates[:-1, :-1, :]
            front_outboard_vertices = wing_coordinates[:-1, 1:, :]
            back_inboard_vertices = wing_coordinates[1:, :-1, :]
            back_outboard_vertices = wing_coordinates[1:, 1:, :]

            # Create a boolean array of whether or not each point is a trailing edge
            is_trailing_edge_this_wing = np.vstack((
                np.zeros((wing_coordinates.shape[0] - 2, wing_coordinates.shape[1] - 1), dtype=bool),
                np.ones((1, wing_coordinates.shape[1] - 1), dtype=bool)
            ))

            # Calculate the colocation points
            colocation_points = (
                    0.25 * (front_inboard_vertices + front_outboard_vertices) / 2 +
                    0.75 * (back_inboard_vertices + back_outboard_vertices) / 2
            )

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
            wing_normals = np.reshape(wing_normals, (-1, 3), order='F')
            inboard_vortex_points = np.reshape(inboard_vortex_points, (-1, 3), order='F')
            outboard_vortex_points = np.reshape(outboard_vortex_points, (-1, 3), order='F')
            front_inboard_vertices = np.reshape(front_inboard_vertices, (-1, 3), order='F')
            front_outboard_vertices = np.reshape(front_outboard_vertices, (-1, 3), order='F')
            back_inboard_vertices = np.reshape(back_inboard_vertices, (-1, 3), order='F')
            back_outboard_vertices = np.reshape(back_outboard_vertices, (-1, 3), order='F')
            is_trailing_edge_this_wing = np.reshape(is_trailing_edge_this_wing, (-1), order='F')

            c = np.vstack((c, colocation_points))
            n = np.vstack((n, wing_normals))
            lv = np.vstack((lv, inboard_vortex_points))
            rv = np.vstack((rv, outboard_vortex_points))
            front_left_vertices = np.vstack((front_left_vertices, front_inboard_vertices))
            front_right_vertices = np.vstack((front_right_vertices, front_outboard_vertices))
            back_left_vertices = np.vstack((back_left_vertices, back_inboard_vertices))
            back_right_vertices = np.vstack((back_right_vertices, back_outboard_vertices))
            is_trailing_edge = np.hstack((is_trailing_edge, is_trailing_edge_this_wing))

            if wing.symmetric:
                inboard_vortex_points = reflect_over_XZ_plane(inboard_vortex_points)
                outboard_vortex_points = reflect_over_XZ_plane(outboard_vortex_points)
                colocation_points = reflect_over_XZ_plane(colocation_points)
                wing_normals = reflect_over_XZ_plane(wing_normals)
                front_inboard_vertices = reflect_over_XZ_plane(front_inboard_vertices)
                front_outboard_vertices = reflect_over_XZ_plane(front_outboard_vertices)
                back_inboard_vertices = reflect_over_XZ_plane(back_inboard_vertices)
                back_outboard_vertices = reflect_over_XZ_plane(back_outboard_vertices)

                c = np.vstack((c, colocation_points))
                n = np.vstack((n, wing_normals))
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
        if self.verbose: print("Calculating the colocation influence matrix...")

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
        if self.verbose: print("Calculating the vortex center influence matrix...")
        self.vortex_centers = (
                                      self.lv + self.rv) / 2  # location of all vortex centers, where the near-field force is assumed to act

        # Redoing the AIC calculation, but using vortex center points instead of colocation points
        # Python Mode
        self.Vij_centers = self.calculate_Vij(self.vortex_centers)
        # Numba Mode
        # self.Vij_centers = self.calculate_Vij_jit(self.vortex_centers, self.lv, self.rv)

        # # LU Decomposition on AIC
        # -------------------------
        if self.verbose: print("LU factorizing the AIC matrix...")
        self.lu, self.piv = sp_linalg.lu_factor(self.AIC)

    def setup_operating_point(self):

        if self.verbose: print("Calculating the freestream influence...")
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
        if self.verbose: print("Calculating vortex strengths...")
        self.vortex_strengths = sp_linalg.lu_solve((self.lu, self.piv), -self.freestream_influences)

    def calculate_forces(self):
        # # Calculate Near-Field Forces and Moments
        # -----------------------------------------
        # Governing Equation: The force on a straight, small vortex filament is F = rho * V x l * gamma,
        # where rho is density, V is the velocity vector, x is the cross product operator,
        # l is the vector of the filament itself, and gamma is the circulation.

        if self.verbose: print("Calculating forces on each panel...")
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

        # Calculate Fi_geometry, the force on the ith panel. Note that this is in GEOMETRY AXES,
        # not WIND AXES or BODY AXES.
        density = self.op_point.density
        Vi_cross_li = np.cross(Vi, self.li, axis=1)
        vortex_strengths_expanded = np.expand_dims(self.vortex_strengths, axis=1)
        self.Fi_geometry = density * Vi_cross_li * vortex_strengths_expanded

        # Calculate total forces and moments
        if self.verbose: print("Calculating total forces and moments...")
        self.Ftotal_geometry = np.sum(self.Fi_geometry,
                                      axis=0)  # Remember, this is in GEOMETRY AXES, not WIND AXES or BODY AXES.
        # if self.verbose: print("Total aerodynamic forces (geometry axes): ", self.Ftotal_geometry)

        self.Ftotal_wind = np.transpose(self.op_point.compute_rotation_matrix_wind_to_geometry()) @ self.Ftotal_geometry
        # if self.verbose: print("Total aerodynamic forces (wind axes):", self.Ftotal_wind)

        self.Mtotal_geometry = np.sum(np.cross(self.vortex_centers - self.airplane.xyz_ref, self.Fi_geometry), axis=0)
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

        if self.verbose: print("\nForces\n-----")
        if self.verbose: print("CL: ", self.CL)
        if self.verbose: print("CDi: ", self.CDi)
        if self.verbose: print("CY: ", self.CY)
        if self.verbose: print("CL/CDi: ", self.CL_over_CDi)
        if self.verbose: print("\nMoments\n-----")
        if self.verbose: print("Cl: ", self.Cl)
        if self.verbose: print("Cm: ", self.Cm)
        if self.verbose: print("Cn: ", self.Cn)

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
        # a: Vector from all colocation points to all horseshoe vortex left  vertices, NxNx3.
        #   # First index is colocation point #, second is vortex #, and third is xyz. N=n_panels
        # b: Vector from all colocation points to all horseshoe vortex right vertices, NxNx3.
        #   # First index is colocation point #, second is vortex #, and third is xyz. N=n_panels
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

    def calculate_streamlines(self):
        # Calculates streamlines eminating from the trailing edges of all surfaces.
        # "streamlines" is a MxNx3 array, where M is the index of the streamline number,
        # N is the index of the timestep, and the last index is xyz

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
            update_amount = update_amount / np.expand_dims(np.linalg.norm(update_amount, axis=1), axis=1)
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
        # Old drawing routine, will be deprecated at some point. Use draw() instead.
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


class vlm2(AeroProblem):
    # Vortex-Lattice Method aerodynamics code written from the ground up with lessons learned from writing VLM1.
    # Should eventually eclipse VLM1 in performance and render it obsolete.
    #
    # Notable improvements over VLM1:
    #   # Specifically written to be reverse-mode-AD-compatible at every step
    #   # Supports control surfaces
    #   # Supports bodies in quasi-steady rotation (nonzero p, q, and r)
    #   # Supports calculation of stability derivatives
    #   # Vortex lattice follows the mean camber line for higher accuracy (though control deflections are done by rotating normals)
    #   # TODO: Takes advantage of the connectivity of the vortex lattice to speed up calculate_Vij() by almost exactly 2x
    #   # TODO: calculate_Vij() is parallelized, one core per wing
    #
    # Usage:
    #   # Set up a problem using the syntax in the AeroProblem constructor (e.g. "vlm2(airplane = a, op_point = op)" for some Airplane a and OperatingPoint op)
    #   # Call vlm2.run() to run the problem.
    #   # Access results in the command line, or through properties of the vlm2 class.
    #   #   # In a future update, this will be done through a standardized AeroData class.

    @profile
    def run(self, verbose=True):
        # Runs a point analysis at the specified op-point.
        self.verbose = verbose

        if self.verbose: print("Running VLM2 calculation...")

        self.check_geometry()
        self.make_panels()
        self.setup_geometry()
        self.setup_operating_point()
        self.calculate_vortex_strengths()
        self.calculate_forces()

        if self.verbose: print("VLM2 calculation complete!")

    def run_stability(self, verbose=True):
        # Runs a stability analysis about the specified op-point.
        # TODO make this function
        self.verbose = verbose

    def check_geometry(self):
        # Make sure things are sensible
        pass  # TODO make this function

    def make_panels(self):
        # Creates self.mcl_coordinates_structured_list and self.wing_mcl_normals.

        if self.verbose: print("Meshing...")

        self.mcl_coordinates_structured_list = []  # List of numpy arrays
        self.normals_structured_list = []  # List of numpy arrays

        for wing_num in range(len(self.airplane.wings)):
            # Things we want for each wing (where M is the number of chordwise panels, N is the number of spanwise panels)
            # # mcl_coordinates_structured_list: M+1 x N+1 x 3; corners of every panel.
            # # normals_structured_list: M x N x 3; normal direction of each panel

            # Get the wing
            wing = self.airplane.wings[wing_num]

            # Define number of chordwise points
            n_chordwise_coordinates = wing.vlm_chordwise_panels + 1

            # Get the chordwise coordinates
            if wing.vlm_chordwise_spacing == 'uniform':
                nondim_chordwise_coordinates = np.linspace(0, 1, n_chordwise_coordinates)
            elif wing.vlm_chordwise_spacing == 'cosine':
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

            # -----------------------------------------------------
            ## Get directions for transforming 2D airfoil data to 3D
            # First, project quarter chords onto YZ plane and normalize.
            section_quarter_chords_proj = (section_quarter_chords[:, 1:] /
                                           np.expand_dims(np.linalg.norm(section_quarter_chords[:, 1:], axis=1), axis=1)
                                           )  # Nx2 array of quarter-chord vectors projected onto YZ plane
            section_quarter_chords_proj = np.hstack(
                (np.zeros((section_quarter_chords_proj.shape[0], 1)), section_quarter_chords_proj)
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

            # -----------------------------------------------------
            ## Get the coordinates of each xsec's airfoil's mean camber line in global coordinates
            # Goal: create xsec_mcl_coordinates, a MxNx3 array of the mean camber line points of each xsec.
            # First index is chordwise point number, second index is xsec number, and third index is xyz.

            # Get the scaling factor (airfoils at dihedral breaks need to be "taller" to compensate)
            xsec_scaling_factor = 1 / np.sqrt((
                                                      1 + np.sum(
                                                  section_quarter_chords_proj[1:, :] * section_quarter_chords_proj[:-1,
                                                                                       :], axis=1
                                              )
                                              ) / 2
                                              )
            xsec_scaling_factor = np.hstack((1, xsec_scaling_factor, 1))
            xsec_camber = np.empty((n_chordwise_coordinates, 0))  # MxN array of camber amounts.
            # First index is chordwise point number, second index is xsec number.
            for xsec in wing.xsecs:
                camber = xsec.airfoil.get_camber_at_chord_fraction(
                    nondim_chordwise_coordinates)  # 1D array of normal directions
                camber = np.expand_dims(camber, axis=1)
                xsec_camber = np.hstack((xsec_camber, camber))

            xsec_mcl_coordinates = (xsec_xyz_le +
                                    xsec_local_back * np.expand_dims(xsec_chord, axis=2) * np.expand_dims(
                        np.expand_dims(nondim_chordwise_coordinates, 1), 2) +
                                    xsec_local_up * np.expand_dims(xsec_chord * xsec_scaling_factor,
                                                                   axis=2) * np.expand_dims(xsec_camber, 2)
                                    )

            # -----------------------------------------------------
            # Interpolate the coordinates between xsecs
            # Goal is to make mcl_coordinates_structured_list
            wing_mcl_coordinates = np.empty((n_chordwise_coordinates, 0, 3))  # MxNx3 of all coordinates on the wing.
            # First index is chordwise point #, second index is spanwise point #, third is xyz.

            for section_num in range(len(wing.xsecs) - 1):
                # Define the relevant cross section
                xsec = wing.xsecs[section_num]

                # Define number of spanwise points
                n_spanwise_coordinates = xsec.vlm_spanwise_panels + 1

                # Get the spanwise coordinates
                if xsec.vlm_spanwise_spacing == 'uniform':
                    nondim_spanwise_coordinates = np.linspace(0, 1, n_spanwise_coordinates)
                elif xsec.vlm_spanwise_spacing == 'cosine':
                    nondim_spanwise_coordinates = cosspace(n_points=n_spanwise_coordinates)
                else:
                    raise Exception("Bad value of section.vlm_spanwise_spacing!")

                # If it's not the last xsec, eliminate the last nondim spanwise coordinate to prevent duplicates
                is_last_section = section_num == len(wing.xsecs) - 2
                if not is_last_section:
                    nondim_spanwise_coordinates = nondim_spanwise_coordinates[:-1]

                section_mcl_coordinates = (
                        np.expand_dims((1 - nondim_spanwise_coordinates), 2) * np.expand_dims(
                    xsec_mcl_coordinates[:, section_num, :], 1) +
                        np.expand_dims(nondim_spanwise_coordinates, 2) * np.expand_dims(
                    xsec_mcl_coordinates[:, section_num + 1, :], 1)
                )  # TODO this is not strictly speaking correct, only true in the limit of small twist angles.
                wing_mcl_coordinates = np.hstack((wing_mcl_coordinates, section_mcl_coordinates))

            # -----------------------------------------------------
            ## Append mean camber line data to vlm2 data list
            self.mcl_coordinates_structured_list.append(wing_mcl_coordinates)
            if wing.symmetric:
                wing_mcl_coordinates_sym = reflect_over_XZ_plane(wing_mcl_coordinates)
                wing_mcl_coordinates_sym = np.fliplr(wing_mcl_coordinates_sym)
                self.mcl_coordinates_structured_list.append(wing_mcl_coordinates_sym)

            # -----------------------------------------------------
            ## Get the normal directions of each xsec's airfoil in nondimensional coordinates
            # Goal: create nondim_xsec_normals, a MxNx2 array of the normal direction of each xsec.
            # First index is chordwise point number, second index is xsec number, and third index is xyz.

            nondim_xsec_normals = np.empty(
                (wing.vlm_chordwise_panels, 0, 2))  # MxNx2 of airfoil normals in local xsec coordinates.
            # First index is chordwise point number, second index is xsec number, and third is LOCAL xy.
            nondim_colocation_coordinates = 0.25 * nondim_chordwise_coordinates[
                                                   :-1] + 0.75 * nondim_chordwise_coordinates[1:]
            for xsec in wing.xsecs:
                nondim_normals = xsec.airfoil.get_normal_direction_at_chord_fraction(nondim_colocation_coordinates)
                nondim_normals = np.expand_dims(nondim_normals, 1)
                nondim_xsec_normals = np.hstack((nondim_xsec_normals, nondim_normals))

            # -----------------------------------------------------
            ## Now, go section-by-section and make the normals while dimensionalizing them.
            # Goal: make normals_structured_list, a MxNx2 array of the normal direction of each panel.
            # First index is chordwise point number, second index is spanwise point number, and third index is xyz.

            wing_normals = np.empty((wing.vlm_chordwise_panels, 0, 3))
            for section_num in range(len(wing.xsecs) - 1):
                # Define the relevant cross section
                xsec = wing.xsecs[section_num]

                # Define number of spanwise points
                n_spanwise_coordinates = xsec.vlm_spanwise_panels + 1

                # Get the spanwise coordinates
                if xsec.vlm_spanwise_spacing == 'uniform':
                    nondim_spanwise_coordinates = np.linspace(0, 1, n_spanwise_coordinates)
                elif xsec.vlm_spanwise_spacing == 'cosine':
                    nondim_spanwise_coordinates = cosspace(n_points=n_spanwise_coordinates)
                else:
                    raise Exception("Bad value of section.vlm_spanwise_spacing!")

                # If it's not the last xsec, eliminate the last nondim spanwise coordinate to prevent duplicates
                is_last_section = section_num == len(wing.xsecs) - 2
                nondim_spanwise_coordinates = (nondim_spanwise_coordinates[1:] + nondim_spanwise_coordinates[:-1]) / 2

                # Get local xsec directions
                # (note: different than xsec_local_back, xsec_local_normal, and xsec_local_up, since these are unaffected by dihedral breaks)
                # Also, these have control surface deflections baked in.
                inner_xsec_back = xsec_local_back[section_num]
                outer_xsec_back = xsec_local_back[section_num + 1]
                section_normal = section_quarter_chords_proj[section_num]
                inner_xsec_up = np.cross(inner_xsec_back, section_normal)
                outer_xsec_up = np.cross(outer_xsec_back, section_normal)

                # Do control surface deflections by rotating the local xsec direction
                control_surface_hinge_point_index = np.interp(
                    x=xsec.control_surface_hinge_point,
                    xp=nondim_colocation_coordinates,
                    fp=np.arange(wing.vlm_chordwise_panels)
                )
                deflection_angle = xsec.control_surface_deflection
                rot_matrix = angle_axis_rotation_matrix(
                    angle=np.radians(deflection_angle),
                    axis=section_normal,
                    axis_already_normalized=True
                )
                inner_xsec_back_rotated = np.matmul(rot_matrix, inner_xsec_back)
                outer_xsec_back_rotated = np.matmul(rot_matrix, outer_xsec_back)
                inner_xsec_up_rotated = np.matmul(rot_matrix, inner_xsec_up)
                outer_xsec_up_rotated = np.matmul(rot_matrix, outer_xsec_up)
                if control_surface_hinge_point_index <= 0:  # For some weird reason, your hinge is at the leading edge
                    inner_xsec_backs = inner_xsec_back_rotated * np.ones((wing.vlm_chordwise_panels, 3))
                    outer_xsec_backs = outer_xsec_back_rotated * np.ones((wing.vlm_chordwise_panels, 3))
                    inner_xsec_ups = inner_xsec_up_rotated * np.ones((wing.vlm_chordwise_panels, 3))
                    outer_xsec_ups = outer_xsec_up_rotated * np.ones((wing.vlm_chordwise_panels, 3))
                elif control_surface_hinge_point_index >= wing.vlm_chordwise_panels:  # For some weird reason, your hinge is at the trailing edge
                    inner_xsec_backs = inner_xsec_back * np.ones((wing.vlm_chordwise_panels, 3))
                    outer_xsec_backs = outer_xsec_back * np.ones((wing.vlm_chordwise_panels, 3))
                    inner_xsec_ups = inner_xsec_up * np.ones((wing.vlm_chordwise_panels, 3))
                    outer_xsec_ups = outer_xsec_up * np.ones((wing.vlm_chordwise_panels, 3))
                else:  # Normal cases, where your hinge isn't at either the leading or trailing edges
                    last_unmodified_index = np.int(np.floor(control_surface_hinge_point_index))
                    fraction_to_modify = 1 - (control_surface_hinge_point_index - last_unmodified_index)
                    rot_matrix = angle_axis_rotation_matrix(
                        angle=np.radians(xsec.control_surface_deflection * fraction_to_modify),
                        axis=section_normal,
                        axis_already_normalized=True
                    )
                    inner_xsec_back_semirotated = np.matmul(rot_matrix, inner_xsec_back)
                    outer_xsec_back_semirotated = np.matmul(rot_matrix, outer_xsec_back)
                    inner_xsec_up_semirotated = np.matmul(rot_matrix, inner_xsec_up)
                    outer_xsec_up_semirotated = np.matmul(rot_matrix, outer_xsec_up)

                    inner_xsec_backs = np.vstack((
                        np.tile(inner_xsec_back, reps=(last_unmodified_index, 1)),
                        inner_xsec_back_semirotated,
                        np.tile(inner_xsec_back_rotated,
                                reps=(wing.vlm_chordwise_panels - last_unmodified_index - 1, 1))
                    ))
                    inner_xsec_ups = np.vstack((
                        np.tile(inner_xsec_up, reps=(last_unmodified_index, 1)),
                        inner_xsec_up_semirotated,
                        np.tile(inner_xsec_up_rotated,
                                reps=(wing.vlm_chordwise_panels - last_unmodified_index - 1, 1))
                    ))
                    outer_xsec_backs = np.vstack((
                        np.tile(outer_xsec_back, reps=(last_unmodified_index, 1)),
                        outer_xsec_back_semirotated,
                        np.tile(outer_xsec_back_rotated,
                                reps=(wing.vlm_chordwise_panels - last_unmodified_index - 1, 1))
                    ))
                    outer_xsec_ups = np.vstack((
                        np.tile(outer_xsec_up, reps=(last_unmodified_index, 1)),
                        outer_xsec_up_semirotated,
                        np.tile(outer_xsec_up_rotated,
                                reps=(wing.vlm_chordwise_panels - last_unmodified_index - 1, 1))
                    ))

                # Get xsec normals
                inner_xsec_normals = (
                        np.expand_dims(nondim_xsec_normals[:, section_num, 0], 1) * inner_xsec_backs +
                        np.expand_dims(nondim_xsec_normals[:, section_num, 1], 1) * inner_xsec_ups
                )  # Nx3 array, where first index is the chordwise point number and second is xyz
                outer_xsec_normals = (
                        np.expand_dims(nondim_xsec_normals[:, section_num + 1, 0], 1) * outer_xsec_backs +
                        np.expand_dims(nondim_xsec_normals[:, section_num + 1, 1], 1) * outer_xsec_ups
                )  # Nx3 array, where first index is the chordwise point number and second is xyz

                # Interpolate between xsec normals
                section_normals = (
                        np.expand_dims((1 - nondim_spanwise_coordinates), 2) * np.expand_dims(inner_xsec_normals, 1) +
                        np.expand_dims(nondim_spanwise_coordinates, 2) * np.expand_dims(outer_xsec_normals, 1)
                )  # TODO this is not strictly speaking correct, only true in the limit of small twist angles.

                # Normalize
                section_normals = section_normals / np.expand_dims(np.linalg.norm(section_normals, axis=2),
                                                                   2)  # TODO This step is not necessary if I fix the interpolate step just prior to this

                # Append
                wing_normals = np.hstack((wing_normals, section_normals))

            self.normals_structured_list.append(wing_normals)

            # -----------------------------------------------------
            ## Symmetry for normals
            if wing.symmetric:
                if wing.has_symmetric_control_surfaces():
                    self.normals_structured_list.append(np.fliplr(reflect_over_XZ_plane(wing_normals)))
                else:
                    # Unfortunately, you kinda have to redo the last mess...
                    # -----------------------------------------------------
                    ## Now, go section-by-section and make the normals while dimensionalizing them.
                    # Goal: make normals_structured_list, a MxNx2 array of the normal direction of each panel.
                    # First index is chordwise point number, second index is spanwise point number, and third index is xyz.

                    wing_normals = np.empty((wing.vlm_chordwise_panels, 0, 3))
                    for section_num in range(len(wing.xsecs) - 1):
                        # Define the relevant cross section
                        xsec = wing.xsecs[section_num]

                        # Define number of spanwise points
                        n_spanwise_coordinates = xsec.vlm_spanwise_panels + 1

                        # Get the spanwise coordinates
                        if xsec.vlm_spanwise_spacing == 'uniform':
                            nondim_spanwise_coordinates = np.linspace(0, 1, n_spanwise_coordinates)
                        elif xsec.vlm_spanwise_spacing == 'cosine':
                            nondim_spanwise_coordinates = cosspace(n_points=n_spanwise_coordinates)
                        else:
                            raise Exception("Bad value of section.vlm_spanwise_spacing!")

                        # If it's not the last xsec, eliminate the last nondim spanwise coordinate to prevent duplicates
                        is_last_section = section_num == len(wing.xsecs) - 2
                        nondim_spanwise_coordinates = (nondim_spanwise_coordinates[1:] + nondim_spanwise_coordinates[
                                                                                         :-1]) / 2

                        # Get local xsec directions
                        # (note: different than xsec_local_back, xsec_local_normal, and xsec_local_up, since these are unaffected by dihedral breaks)
                        # Also, these have control surface deflections baked in.
                        inner_xsec_back = xsec_local_back[section_num]
                        outer_xsec_back = xsec_local_back[section_num + 1]
                        section_normal = section_quarter_chords_proj[section_num]
                        inner_xsec_up = np.cross(inner_xsec_back, section_normal)
                        outer_xsec_up = np.cross(outer_xsec_back, section_normal)

                        # Do control surface deflections by rotating the local xsec direction
                        control_surface_hinge_point_index = np.interp(
                            x=xsec.control_surface_hinge_point,
                            xp=nondim_colocation_coordinates,
                            fp=np.arange(wing.vlm_chordwise_panels)
                        )
                        deflection_angle = xsec.control_surface_deflection
                        if xsec.control_surface_type == "asymmetric":
                            deflection_angle = -deflection_angle
                        rot_matrix = angle_axis_rotation_matrix(
                            angle=np.radians(deflection_angle),
                            axis=section_normal,
                            axis_already_normalized=True
                        )
                        inner_xsec_back_rotated = np.matmul(rot_matrix, inner_xsec_back)
                        outer_xsec_back_rotated = np.matmul(rot_matrix, outer_xsec_back)
                        inner_xsec_up_rotated = np.matmul(rot_matrix, inner_xsec_up)
                        outer_xsec_up_rotated = np.matmul(rot_matrix, outer_xsec_up)
                        if control_surface_hinge_point_index <= 0:  # For some weird reason, your hinge is at the leading edge
                            inner_xsec_backs = inner_xsec_back_rotated * np.ones((wing.vlm_chordwise_panels, 3))
                            outer_xsec_backs = outer_xsec_back_rotated * np.ones((wing.vlm_chordwise_panels, 3))
                            inner_xsec_ups = inner_xsec_up_rotated * np.ones((wing.vlm_chordwise_panels, 3))
                            outer_xsec_ups = outer_xsec_up_rotated * np.ones((wing.vlm_chordwise_panels, 3))
                        elif control_surface_hinge_point_index >= wing.vlm_chordwise_panels:  # For some weird reason, your hinge is at the trailing edge
                            inner_xsec_backs = inner_xsec_back * np.ones((wing.vlm_chordwise_panels, 3))
                            outer_xsec_backs = outer_xsec_back * np.ones((wing.vlm_chordwise_panels, 3))
                            inner_xsec_ups = inner_xsec_up * np.ones((wing.vlm_chordwise_panels, 3))
                            outer_xsec_ups = outer_xsec_up * np.ones((wing.vlm_chordwise_panels, 3))
                        else:  # Normal cases, where your hinge isn't at either the leading or trailing edges
                            last_unmodified_index = np.int(np.floor(control_surface_hinge_point_index))
                            fraction_to_modify = 1 - (control_surface_hinge_point_index - last_unmodified_index)
                            rot_matrix = angle_axis_rotation_matrix(
                                angle=np.radians(xsec.control_surface_deflection * fraction_to_modify),
                                axis=section_normal,
                                axis_already_normalized=True
                            )
                            inner_xsec_back_semirotated = np.matmul(rot_matrix, inner_xsec_back)
                            outer_xsec_back_semirotated = np.matmul(rot_matrix, outer_xsec_back)
                            inner_xsec_up_semirotated = np.matmul(rot_matrix, inner_xsec_up)
                            outer_xsec_up_semirotated = np.matmul(rot_matrix, outer_xsec_up)

                            inner_xsec_backs = np.vstack((
                                np.tile(inner_xsec_back, reps=(last_unmodified_index, 1)),
                                inner_xsec_back_semirotated,
                                np.tile(inner_xsec_back_rotated,
                                        reps=(wing.vlm_chordwise_panels - last_unmodified_index - 1, 1))
                            ))
                            inner_xsec_ups = np.vstack((
                                np.tile(inner_xsec_up, reps=(last_unmodified_index, 1)),
                                inner_xsec_up_semirotated,
                                np.tile(inner_xsec_up_rotated,
                                        reps=(wing.vlm_chordwise_panels - last_unmodified_index - 1, 1))
                            ))
                            outer_xsec_backs = np.vstack((
                                np.tile(outer_xsec_back, reps=(last_unmodified_index, 1)),
                                outer_xsec_back_semirotated,
                                np.tile(outer_xsec_back_rotated,
                                        reps=(wing.vlm_chordwise_panels - last_unmodified_index - 1, 1))
                            ))
                            outer_xsec_ups = np.vstack((
                                np.tile(outer_xsec_up, reps=(last_unmodified_index, 1)),
                                outer_xsec_up_semirotated,
                                np.tile(outer_xsec_up_rotated,
                                        reps=(wing.vlm_chordwise_panels - last_unmodified_index - 1, 1))
                            ))

                        # Get xsec normals
                        inner_xsec_normals = (
                                np.expand_dims(nondim_xsec_normals[:, section_num, 0], 1) * inner_xsec_backs +
                                np.expand_dims(nondim_xsec_normals[:, section_num, 1], 1) * inner_xsec_ups
                        )  # Nx3 array, where first index is the chordwise point number and second is xyz
                        outer_xsec_normals = (
                                np.expand_dims(nondim_xsec_normals[:, section_num + 1, 0], 1) * outer_xsec_backs +
                                np.expand_dims(nondim_xsec_normals[:, section_num + 1, 1], 1) * outer_xsec_ups
                        )  # Nx3 array, where first index is the chordwise point number and second is xyz

                        # Interpolate between xsec normals
                        section_normals = (
                                np.expand_dims((1 - nondim_spanwise_coordinates), 2) * np.expand_dims(
                            inner_xsec_normals, 1) +
                                np.expand_dims(nondim_spanwise_coordinates, 2) * np.expand_dims(outer_xsec_normals, 1)
                        )  # TODO this is not strictly speaking correct, only true in the limit of small twist angles.

                        # Normalize
                        section_normals = section_normals / np.expand_dims(np.linalg.norm(section_normals, axis=2),
                                                                           2)  # TODO This step is not necessary if I fix the interpolate step just prior to this

                        # Append
                        wing_normals = np.hstack((wing_normals, section_normals))

                    self.normals_structured_list.append(np.flip(reflect_over_XZ_plane(wing_normals), axis=1))

        if self.verbose: print("Meshing complete!")
        # -----------------------------------------------------
        # Review of the important things that have been done up to this point:
        # * We made mcl_coordinates_structured_list, a MxNx3 array describing a structured quadrilateral mesh of the wing's mean camber surface.
        #   * For reference: first index is chordwise coordinate, second index is spanwise coordinate, and third index is xyz.
        # * We made normals_structured_list, a MxNx3 array describing the normal direction of the mean camber surface at the colocation point.
        #   * For reference: first index is chordwise coordinate, second index is spanwise coordinate, and third index is xyz.
        #   * Takes into account control surface deflections
        # * Both mcl_coordinates_structured_list and normals_structured_list have been appended to lists of ndarrays within the vlm2 class,
        #   accessible at self.mcl_coordinates_structured_list and self.normals_structured_list, respectively.
        # * Control surface handling:
        #   * Control surfaces are implemented into normal directions as intended.
        # * Symmetry handling:
        #   * All symmetric wings have been split into separate halves.
        #   * All wing halves have their spanwise coordinates labeled from the left side of the airplane to the right.
        #   * Control surface deflection symmetry has been handled; this is encoded into the normal directions.
        # * And best of all, it's all verified to be reverse-mode AD compatible!!!

        # -----------------------------------------------------
        ## Now, just post-process them to get the colocation points and vortex center points.
        self.n_wings = len(self.mcl_coordinates_structured_list)  # Good to know

        self.front_left_vertices_list=[]
        self.front_right_vertices_list=[]
        self.back_left_vertices_list=[]
        self.back_right_vertices_list=[]

        self.vortex_left_list = []
        self.vortex_right_list = []

        self.colocations_list = []

        self.normals_list = []

        for wing_num in range(self.n_wings):
            wing_front_left_vertices = self.mcl_coordinates_structured_list[wing_num][:-1, :-1, :]
            wing_front_right_vertices = self.mcl_coordinates_structured_list[wing_num][:-1, 1:, :]
            wing_back_left_vertices = self.mcl_coordinates_structured_list[wing_num][1:, :-1, :]
            wing_back_right_vertices = self.mcl_coordinates_structured_list[wing_num][1:, 1:, :]

            self.front_left_vertices_list.append(np.reshape(wing_front_left_vertices,(-1,3)))
            self.front_right_vertices_list.append(np.reshape(wing_front_right_vertices,(-1,3)))
            self.back_left_vertices_list.append(np.reshape(wing_back_left_vertices,(-1,3)))
            self.back_right_vertices_list.append(np.reshape(wing_back_right_vertices,(-1,3)))

            self.colocations_list.append(
                np.reshape((
                        0.5 * (0.25 * wing_front_left_vertices +  # Left front
                               0.75 * wing_back_left_vertices) +  # Left back
                        0.5 * (0.25 * wing_front_right_vertices +  # Right front
                               0.75 * wing_back_right_vertices)  # Right back
                ),
                    (-1, 3)
                )
            )
            self.vortex_left_list.append(
                np.reshape(
                    0.75 * wing_front_left_vertices +  # Left front
                    0.25 * wing_back_left_vertices,  # Left back
                    (-1, 3))
            )
            self.vortex_right_list.append(
                np.reshape(
                    0.75 * wing_front_right_vertices +  # Right front
                    0.25 * wing_back_right_vertices,  # Right back
                    (-1, 3))
            )
            self.normals_list.append(
                np.reshape(self.normals_structured_list[wing_num], (-1, 3))
            )

        self.front_left_vertices_unrolled = np.vstack(self.front_left_vertices_list)
        self.front_right_vertices_unrolled = np.vstack(self.front_right_vertices_list)
        self.back_left_vertices_unrolled = np.vstack(self.back_left_vertices_list)
        self.back_right_vertices_unrolled = np.vstack(self.back_right_vertices_list)
        
        self.colocations_unrolled = np.vstack(self.colocations_list)
        self.vortex_left_unrolled = np.vstack(self.vortex_left_list)
        self.vortex_right_unrolled = np.vstack(self.vortex_right_list)
        self.vortex_centers_unrolled = (self.vortex_left_unrolled + self.vortex_right_unrolled) / 2
        self.normals_unrolled = np.vstack(self.normals_list)

        self.n_panels = len(self.normals_unrolled)  # Also good to know

        ## For debugging only # TODO delete later

    def setup_geometry(self):
        # # Calculate AIC matrix
        # ----------------------
        if self.verbose: print("Calculating the colocation influence matrix...")
        self.Vij_colocations = self.calculate_Vij(self.colocations_unrolled)
        # Vij_colocations: [points, vortices, xyz]
        # n: [points, xyz]

        normals_expanded = np.expand_dims(self.normals_unrolled, 1)

        # AIC = (Vij * normal vectors)
        self.AIC = np.sum(
            self.Vij_colocations * normals_expanded,
            axis=2
        )

        # # Calculate Vij at vortex centers for force calculation
        # -------------------------------------------------------
        if self.verbose: print("Calculating the vortex center influence matrix...")
        self.Vij_centers = self.calculate_Vij(self.vortex_centers_unrolled)

        # # LU Decomposition on AIC
        # -------------------------
        # Unfortunately, I don't think we can use sp_linalg.lu_factor with autograd, so we'll have to do a direct solve
        # for every op-point instead of saving an LU-factorization and reusing it.
        # This isn't the worst, though, since the solution time is very small compared to the AIC calculation time, and
        # autograd gives us good gradients to use for op-point trimming and other things. So that's nice. I guess.
        # The long-term solution here would be to write a vector-jacobian product in autograd for lu_factor and lu_solve.

        # Old code
        # if self.verbose: print("LU factorizing the AIC matrix...")
        # self.lu, self.piv = sp_linalg.lu_factor(self.AIC) # TODO consider whether lu_factor is possible w autograd

    def setup_operating_point(self):  # TODO hasn't been checked yet

        if self.verbose: print("Calculating the freestream influence...")
        self.steady_freestream_velocity = self.op_point.compute_freestream_velocity_geometry_axes() * np.ones(
            (self.n_panels, 1))  # Direction the wind is GOING TO, in geometry axes coordinates
        self.rotation_freestream_velocities = self.op_point.compute_rotation_velocity_geometry_axes(self.colocations_unrolled)


        #np.zeros((self.n_panels, 3))  # TODO Make this actually be the rotational velocity

        self.freestream_velocities = self.steady_freestream_velocity + self.rotation_freestream_velocities  # Nx3, represents the freestream velocity at each panel colocation point (c)

        self.freestream_influences = np.sum(self.freestream_velocities * self.normals_unrolled, axis=1)

    def calculate_vortex_strengths(self):
        # # Calculate Vortex Strengths
        # ----------------------------
        # Governing Equation: AIC @ Gamma + freestream_influence = 0
        if self.verbose: print("Calculating vortex strengths...")
        self.vortex_strengths = np.linalg.solve(self.AIC, -self.freestream_influences)

    def calculate_forces(self):
        # # Calculate Near-Field Forces and Moments
        # -----------------------------------------
        # Governing Equation: The force on a straight, small vortex filament is F = rho * V x l * gamma,
        # where rho is density, V is the velocity vector, x is the cross product operator,
        # l is the vector of the filament itself, and gamma is the circulation.

        if self.verbose: print("Calculating forces on each panel...")
        # Calculate Vi (local velocity at the ith vortex center point)
        Vi_x = self.Vij_centers[:, :, 0] @ self.vortex_strengths + self.freestream_velocities[:, 0]
        Vi_y = self.Vij_centers[:, :, 1] @ self.vortex_strengths + self.freestream_velocities[:, 1]
        Vi_z = self.Vij_centers[:, :, 2] @ self.vortex_strengths + self.freestream_velocities[:, 2]
        Vi_x = np.expand_dims(Vi_x, axis=1)
        Vi_y = np.expand_dims(Vi_y, axis=1)
        Vi_z = np.expand_dims(Vi_z, axis=1)
        Vi = np.hstack((Vi_x, Vi_y, Vi_z))

        # Calculate li, the length of the bound segment of the horseshoe vortex filament
        # self.li = self.rv - self.lv
        li_pieces = []
        for wing_num in range(self.n_wings):
            wing_mcl_coordinates = self.mcl_coordinates_structured_list[wing_num]
            wing_vortex_points = (
                    0.75 * wing_mcl_coordinates[:-1, :, :] +
                    0.25 * wing_mcl_coordinates[1:, :, :]
            )
            li_piece = wing_vortex_points[:, 1:, :] - wing_vortex_points[:, :-1, :]
            li_piece = np.reshape(li_piece, (-1, 3))
            li_pieces.append(li_piece)
        self.li = np.vstack(li_pieces)

        # Calculate Fi_geometry, the force on the ith panel. Note that this is in GEOMETRY AXES,
        # not WIND AXES or BODY AXES.
        density = self.op_point.density
        Vi_cross_li = np.cross(Vi, self.li, axis=1)
        vortex_strengths_expanded = np.expand_dims(self.vortex_strengths, axis=1)
        self.Fi_geometry = density * Vi_cross_li * vortex_strengths_expanded

        # Calculate total forces and moments
        if self.verbose: print("Calculating total forces and moments...")
        self.Ftotal_geometry = np.sum(self.Fi_geometry,
                                      axis=0)  # Remember, this is in GEOMETRY AXES, not WIND AXES or BODY AXES.
        # if self.verbose: print("Total aerodynamic forces (geometry axes): ", self.Ftotal_geometry)

        self.Ftotal_wind = np.transpose(self.op_point.compute_rotation_matrix_wind_to_geometry()) @ self.Ftotal_geometry
        # if self.verbose: print("Total aerodynamic forces (wind axes):", self.Ftotal_wind)

        self.Mtotal_geometry = np.sum(np.cross(self.vortex_centers_unrolled - self.airplane.xyz_ref, self.Fi_geometry),
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

        if self.verbose: print("\nForces\n-----")
        if self.verbose: print("CL: ", self.CL)
        if self.verbose: print("CDi: ", self.CDi)
        if self.verbose: print("CY: ", self.CY)
        if self.verbose: print("CL/CDi: ", self.CL_over_CDi)
        if self.verbose: print("\nMoments\n-----")
        if self.verbose: print("Cl: ", self.Cl)
        if self.verbose: print("Cm: ", self.Cm)
        if self.verbose: print("Cn: ", self.Cn)

    @profile
    def calculate_Vij_wing_by_wing(self, points):
        # Calculates Vij, the velocity influence matrix (First index is colocation point number, second index is vortex number).
        # points: the list of points (Nx3) to calculate the velocity influence at.
        #
        # a and b are the vectors from the points to the horseshoe vortex corners. For a picture, see Drela's "Flight Vehicle Aerodynamics", pg. 132. Or message me.
        #
        points = np.reshape(points, (-1, 3))  # if it wasn't already, points is now a Nx3 array.
        n_points = len(points)

        # Make Vij for each wing
        Vij_pieces = []
        for wing_num in range(self.n_wings):
            wing_mcl_coordinates = self.mcl_coordinates_structured_list[wing_num]
            wing_vortex_points = (
                    0.75 * wing_mcl_coordinates[:-1, :, :] +
                    0.25 * wing_mcl_coordinates[1:, :, :]
            )
            wing_ab = np.expand_dims(np.expand_dims(points, 1), 2) - wing_vortex_points
            # wing_ab is a 4D array
            # 1st index: point #, 2nd index: chordwise point #, 3rd index: spanwise point #, 4th index: xyz.
            # Exploits the fact that a and b are almost exactly the same.
            wing_ab_shape = wing_ab.shape

            # ----------------------------------------------------
            # Do a bunch of useful arithmetic
            wing_ab_cross_x = np.stack((
                np.zeros((n_points, wing_ab_shape[1], wing_ab_shape[2])),
                wing_ab[:, :, :, 2],
                -wing_ab[:, :, :, 1],
            ),
                axis=3
            )
            wing_ab_dot_x = wing_ab[:, :, :, 0]
            wing_a_cross_b = np.cross(
                wing_ab[:, :, :-1, :],
                wing_ab[:, :, 1:, :],
                axis=3
            )
            wing_a_dot_b = np.einsum('ijkl,ijkl->ijk', wing_ab[:, :, :-1, :], wing_ab[:, :, 1:, :])

            wing_ab_norm = np.linalg.norm(wing_ab, axis=3)
            wing_ab_norm_inv = 1 / wing_ab_norm

            # ----------------------------------------------------
            # Check for the special case where a colocation point is along a vortex filament
            # Find where cross product is near zero, and set the dot product to infinity so that the value of the bound term is zero.
            # TODO: Dear future Peter: this wouldn't be a problem if you had just implemented a nonsingular kernel, you dummy
            bound_vortex_singularity_indices = (
                    np.einsum('ijkl,ijkl->ijk', wing_a_cross_b, wing_a_cross_b)  # norm(a_cross_b) ** 2
                    < 3.0e-16)
            wing_a_dot_b = wing_a_dot_b + bound_vortex_singularity_indices  # Basically, adds 1 to throw off the calculation if the bound term is near zero.
            side_vortex_singularity_indices = (
                    np.einsum('ijkl,ijkl->ijk', wing_ab_cross_x, wing_ab_cross_x)
                    < 3.0e-16
            )
            wing_ab_dot_x = wing_ab_dot_x + side_vortex_singularity_indices

            # ----------------------------------------------------
            # Parse into a and b, not ab.
            wing_a_cross_x = wing_ab_cross_x[:, :, :-1, :]
            wing_b_cross_x = wing_ab_cross_x[:, :, 1:, :]
            wing_a_dot_x = wing_ab_dot_x[:, :, :-1]
            wing_b_dot_x = wing_ab_dot_x[:, :, 1:]
            wing_a_norm = wing_ab_norm[:, :, :-1]
            wing_b_norm = wing_ab_norm[:, :, 1:]
            wing_a_norm_inv = wing_ab_norm_inv[:, :, :-1]
            wing_b_norm_inv = wing_ab_norm_inv[:, :, 1:]

            # ----------------------------------------------------
            # Do reshaping of lots of things into Vij form
            wing_a_cross_b = np.reshape(wing_a_cross_b, (n_points, -1, 3))
            wing_a_cross_x = np.reshape(wing_a_cross_x, (n_points, -1, 3))
            wing_b_cross_x = np.reshape(wing_b_cross_x, (n_points, -1, 3))
            wing_a_dot_b = np.reshape(wing_a_dot_b, (n_points, -1))
            wing_a_dot_x = np.reshape(wing_a_dot_x, (n_points, -1))
            wing_b_dot_x = np.reshape(wing_b_dot_x, (n_points, -1))
            wing_a_norm = np.reshape(wing_a_norm, (n_points, -1))
            wing_b_norm = np.reshape(wing_b_norm, (n_points, -1))
            wing_a_norm_inv = np.reshape(wing_a_norm_inv, (n_points, -1))
            wing_b_norm_inv = np.reshape(wing_b_norm_inv, (n_points, -1))

            # ----------------------------------------------------
            # Do the calculation for this part of Vij
            term1 = (wing_a_norm_inv + wing_b_norm_inv) / (wing_a_norm * wing_b_norm + wing_a_dot_b)
            term2 = (wing_a_norm_inv) / (wing_a_norm - wing_a_dot_x)
            term3 = (wing_b_norm_inv) / (wing_b_norm - wing_b_dot_x)
            term1 = np.expand_dims(term1, 2)
            term2 = np.expand_dims(term2, 2)
            term3 = np.expand_dims(term3, 2)

            Vij_piece = 1 / (4 * np.pi) * (
                    wing_a_cross_b * term1 +
                    wing_a_cross_x * term2 -
                    wing_b_cross_x * term3
            )

            Vij_pieces.append(Vij_piece)

        Vij = np.hstack(Vij_pieces)

        return Vij

    @profile
    def calculate_Vij(self, points):  # TODO finish this or delete this
        # Calculates Vij, the velocity influence matrix (First index is colocation point number, second index is vortex number).
        # points: the list of points (Nx3) to calculate the velocity influence at.

        # Make lv and rv
        left_vortex_points = self.vortex_left_unrolled
        right_vortex_points = self.vortex_right_unrolled

        points = np.reshape(points, (-1, 3))
        n_points = len(points)
        n_vortices = self.n_panels

        # Make a and b vectors.
        # a: Vector from all colocation points to all horseshoe vortex left  vertices, NxNx3.
        #   # First index is colocation point #, second is vortex #, and third is xyz. N=n_panels
        # b: Vector from all colocation points to all horseshoe vortex right vertices, NxNx3.
        #   # First index is colocation point #, second is vortex #, and third is xyz. N=n_panels
        # a[i,j,:] = c[i,:] - lv[j,:]
        # b[i,j,:] = c[i,:] - rv[j,:]
        points = np.expand_dims(points, 1)
        a = points - left_vortex_points
        b = points - right_vortex_points
        # x_hat = np.zeros([n_points, n_vortices, 3])
        # x_hat[:, :, 0] = 1

        # Do some useful arithmetic
        a_cross_b = np.cross(a, b, axis=2)
        a_dot_b = np.einsum('ijk,ijk->ij', a, b)

        a_cross_x = np.dstack((
            np.zeros((n_points, n_vortices)),
            a[:,:,2],
            -a[:,:,1]
        ))
        a_dot_x = a[:, :, 0]

        b_cross_x = np.dstack((
            np.zeros((n_points, n_vortices)),
            b[:,:,2],
            -b[:,:,1]
        ))
        b_dot_x = b[:, :, 0]  # np.sum(b * x_hat,axis=2)

        norm_a = np.linalg.norm(a, axis=2)
        norm_b = np.linalg.norm(b, axis=2)
        norm_a_inv = 1 / norm_a
        norm_b_inv = 1 / norm_b

        # Check for the special case where the colocation point is along the bound vortex leg
        # Find where cross product is near zero, and set the dot product to infinity so that the value of the bound term is zero.
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

    # TODO FIX FROM HERE ON
    def calculate_delta_cp(self):
        # Find the area of each panel ()
        diag1 = self.front_left_vertices_unrolled - self.back_right_vertices_unrolled
        diag2 = self.front_right_vertices_unrolled - self.back_left_vertices_unrolled
        self.areas = np.linalg.norm(np.cross(diag1, diag2, axis=1), axis=1) / 2

        # Calculate panel data
        self.Fi_normal = np.einsum('ij,ij->i', self.Fi_geometry, self.normals_unrolled)
        self.pressure_normal = self.Fi_normal / self.areas
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

    def calculate_streamlines(self):
        # Calculates streamlines eminating from the trailing edges of all surfaces.
        # "streamlines" is a MxNx3 array, where M is the index of the streamline number,
        # N is the index of the timestep, and the last index is xyz

        # Constants
        n_steps = 100  # minimum of 2
        length = 1  # meter

        # Resolution
        length_per_step = length / n_steps
        # dt = length / self.op_point.velocity / n_steps

        # Seed points
        #seed_points = (0.5 * (self.back_left_vertices_unrolled + self.back_right_vertices_unrolled))[self.is_trailing_edge]
        seed_points_list = []
        for wing_num in range(self.n_wings):
            wing_mcl_coordinates = self.mcl_coordinates_structured_list[wing_num]
            wing_te_coordinates = wing_mcl_coordinates[-1,:,:]
            wing_seed_points = (wing_te_coordinates[:-1,:]+wing_te_coordinates[1:,:])/2
            seed_points_list.append(wing_seed_points)
        seed_points = np.vstack(seed_points_list)

        n_streamlines = len(seed_points)

        # Initialize
        streamlines = np.zeros((n_streamlines, n_steps, 3))
        streamlines[:, 0, :] = seed_points

        # Iterate
        for step_num in range(1, n_steps):
            update_amount = self.get_velocity_at_point(streamlines[:, step_num - 1, :])
            update_amount = update_amount * length_per_step / np.expand_dims(np.linalg.norm(update_amount, axis=1), axis=1)
            streamlines[:, step_num, :] = streamlines[:, step_num - 1, :] + update_amount

        self.streamlines = streamlines

    def draw(self,
             draw_delta_cp=True,
             draw_streamlines=True,
             ):

        print("Drawing...")
        #
        # Note: NOT autograd-compatible!


        # Make airplane geometry
        vertices = np.vstack((
            self.front_left_vertices_unrolled,
            self.front_right_vertices_unrolled,
            self.back_right_vertices_unrolled,
            self.back_left_vertices_unrolled
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

            delta_cp_min = -1.5
            delta_cp_max = 1.5

            scalars = np.minimum(np.maximum(self.delta_cp, delta_cp_min), delta_cp_max)
            cmap = plt.cm.get_cmap('viridis')
            plotter.add_mesh(wing_surfaces, scalars=scalars, cmap=cmap, color='tan', show_edges=True,
                             smooth_shading=True)
            plotter.add_scalar_bar(title="Pressure Coefficient Differential", n_labels=5, shadow=True, font_family='arial')

        if draw_streamlines:
            if not hasattr(self, 'streamlines'):
                self.calculate_streamlines()

            for streamline_num in range(len(self.streamlines)):
                plotter.add_lines(self.streamlines[streamline_num, :, :], width=1, color='#50C7C7')

        # Do the plotting
        plotter.show_grid(color='#444444')
        plotter.set_background(color="black")
        plotter.show(cpos=(-1, -1, 1), full_screen=False)


