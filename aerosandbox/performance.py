import autograd.numpy as np
import math
import matplotlib.pyplot as plt
from .plotting import *


class OperatingPoint:
    def __init__(self,
                 density = 1.225,
                 velocity=10,
                 alpha=5,
                 beta=0,
                 p=0, # About the body x-axis
                 q=0, # About the body y-axis
                 r=0, # About the body z-axis
                 ):
        self.density = density
        self.velocity = velocity
        self.alpha = alpha
        self.beta = beta
        self.p = p
        self.q = q
        self.r = r

    def dynamic_pressure(self):
        return 0.5*self.density*self.velocity**2

    def compute_rotation_matrix_wind_to_geometry(self):
        # Computes the 3x3 rotation matrix required to go from wind axes to geometry axes.
        sinalpha = np.sin(np.radians(self.alpha))
        cosalpha = np.cos(np.radians(self.alpha))
        sinbeta = np.sin(np.radians(self.beta))
        cosbeta = np.cos(np.radians(self.beta))

        # r=-1*np.array([
        #     [cosbeta*cosalpha, -sinbeta, cosbeta*sinalpha],
        #     [sinbeta*cosalpha, cosbeta, sinbeta*sinalpha],
        #     [-sinalpha, 0, cosalpha]
        # ])

        eye = np.eye(3)

        alpharotation = np.array([
            [cosalpha, 0, -sinalpha],
            [0, 1, 0],
            [sinalpha, 0, cosalpha]
        ])

        betarotation = np.array([
            [cosbeta, -sinbeta, 0],
            [sinbeta, cosbeta, 0],
            [0, 0, 1]
        ])

        axesflip = np.array([
            [-1, 0, 0],
            [0, 1, 0, ],
            [0, 0, -1]
        ]) # Since in geometry axes, X is downstream by convention, while in wind axes, X is upstream by convetion. Same with Z being up/down respectively.

        r = axesflip @ alpharotation @ betarotation @ eye  # where "@" is the matrix multiplication operator

        return r

    def compute_freestream_direction_geometry_axes(self):
        # Computes the freestream direction (direction the wind is GOING TO) in the geometry axes
        vel_dir_wind = np.array([-1, 0, 0])
        vel_dir_geometry = self.compute_rotation_matrix_wind_to_geometry() @ vel_dir_wind
        return vel_dir_geometry

    def compute_freestream_velocity_geometry_axes(self):
        # Computes the freestream velocity vector (direction the wind is GOING TO) in geometry axes
        return self.compute_freestream_direction_geometry_axes() * self.velocity

    def compute_rotation_velocity_geometry_axes(self, points):
        # Computes the effective velocity due to rotation at a set of points.
        # Input: a Nx3 array of points
        # Output: a Nx3 array of effective velocities
        angular_velocity_vector_geometry_axes = np.array([-self.p, self.q, -self.r]) # signs convert from body axes to geometry axes
        angular_velocity_vector_geometry_axes = np.expand_dims(angular_velocity_vector_geometry_axes, axis = 0)


        rotation_velocity_geometry_axes = np.cross(
            angular_velocity_vector_geometry_axes,
            points,
            axis=1
        )

        rotation_velocity_geometry_axes = -rotation_velocity_geometry_axes # negative sign, since we care about the velocity the WING SEES, not the velocity of the wing.

        return rotation_velocity_geometry_axes