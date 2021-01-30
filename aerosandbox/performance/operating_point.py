import casadi as cas
from aerosandbox.geometry import *
from aerosandbox import Atmosphere
from numpy import pi


class OperatingPoint(AeroSandboxObject):
    def __init__(self,
                 atmosphere: Atmosphere = Atmosphere(altitude=0),
                 velocity: float = 10.,  # m/s
                 alpha: float = 5.,  # In degrees
                 beta: float = 0.,  # In degrees
                 p: float = 0.,  # About the body x-axis, in rad/sec
                 q: float = 0.,  # About the body y-axis, in rad/sec
                 r: float = 0.,  # About the body z-axis, in rad/sec
                 ):
        self.atmosphere = atmosphere
        self.altitude = altitude
        self.velocity = velocity
        self.alpha = alpha
        self.beta = beta
        self.p = p
        self.q = q
        self.r = r

    def dynamic_pressure(self):
        """ Dynamic pressure of the working fluid
        .. math:: p = \frac{\\rho u^2}{2}
        Args:
            self.density (float): Density of the working fluid in .. math:: \frac{kg}{m^3}
            self.velocity (float): Velocity of the working fluid in .. math:: \frac{m}{s}
        Returns:
            float: Dynamic pressure of the working fluid in .. math:: \frac{N}{m^2}
        """
        return 0.5 * self.density * self.velocity ** 2

    def reynolds(self, reference_length):
        """
        Computes a Reynolds number with respect to a given reference length.
        :param reference_length: A reference length you choose [m]
        :return: Reynolds number [unitless]
        """
        density = self.atmosphere.density()
        viscosity = self.atmosphere.dynamic_viscosity()

        return density * self.velocity * reference_length / viscosity

    def mach(self):
        """
        Return the mach number associated with the current flight condition.
        """
        return self.velocity / self.atmosphere.speed_of_sound()

    def compute_rotation_matrix_wind_to_geometry(self):
        # Computes the 3x3 rotation matrix required to go from wind axes to geometry axes.
        sinalpha = np.sin(self.alpha * pi / 180)
        cosalpha = np.cos(self.alpha * pi / 180)
        sinbeta = np.sin(self.beta * pi / 180)
        cosbeta = np.cos(self.beta * pi / 180)

        # r=-1*cas.array([
        #     [cosbeta*cosalpha, -sinbeta, cosbeta*sinalpha],
        #     [sinbeta*cosalpha, cosbeta, sinbeta*sinalpha],
        #     [-sinalpha, 0, cosalpha]
        # ])

        alpharotation = cas.vertcat(
            cas.horzcat(cosalpha, 0, -sinalpha),
            cas.horzcat(0, 1, 0),
            cas.horzcat(sinalpha, 0, cosalpha),
        )

        betarotation = cas.vertcat(
            cas.horzcat(cosbeta, -sinbeta, 0),
            cas.horzcat(sinbeta, cosbeta, 0),
            cas.horzcat(0, 0, 1),
        )

        axesflip = cas.DM([
            [-1, 0, 0],
            [0, 1, 0, ],
            [0, 0, -1]
        ])  # Since in geometry axes, X is downstream by convention, while in wind axes, X is upstream by convetion. Same with Z being up/down respectively.

        eye = cas.DM_eye(3)

        r = axesflip @ alpharotation @ betarotation @ eye  # where "@" is the matrix multiplication operator

        return r

    def compute_freestream_direction_geometry_axes(self):
        # Computes the freestream direction (direction the wind is GOING TO) in the geometry axes
        vel_dir_wind = cas.DM([-1, 0, 0])
        vel_dir_geometry = self.compute_rotation_matrix_wind_to_geometry() @ vel_dir_wind
        return vel_dir_geometry

    def compute_freestream_velocity_geometry_axes(self):
        # Computes the freestream velocity vector (direction the wind is GOING TO) in geometry axes
        return self.compute_freestream_direction_geometry_axes() * self.velocity

    def compute_rotation_velocity_geometry_axes(self, points):
        # Computes the effective velocity due to rotation at a set of points.
        # Input: a Nx3 array of points
        # Output: a Nx3 array of effective velocities
        angular_velocity_vector_geometry_axes = cas.vertcat(
            -self.p, self.q, -self.r)  # signs convert from body axes to geometry axes
        # angular_velocity_vector_geometry_axes = cas.expand_dims(angular_velocity_vector_geometry_axes, axis=0)

        a = angular_velocity_vector_geometry_axes
        b = points

        rotation_velocity_geometry_axes = cas.horzcat(
            a[1] * b[:, 2] - a[2] * b[:, 1],
            a[2] * b[:, 0] - a[0] * b[:, 2],
            a[0] * b[:, 1] - a[1] * b[:, 0]
        )

        rotation_velocity_geometry_axes = -rotation_velocity_geometry_axes  # negative sign, since we care about the velocity the WING SEES, not the velocity of the wing.

        return rotation_velocity_geometry_axes
