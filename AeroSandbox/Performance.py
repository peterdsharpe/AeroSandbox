import autograd.numpy as np
import math
import matplotlib.pyplot as plt
from .Plotting import *


class OperatingPoint:
    def __init__(self,
                 density = 1.225,
                 velocity=10,
                 alpha=5,
                 beta=0,
                 p=0,
                 q=0,
                 r=0,
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
