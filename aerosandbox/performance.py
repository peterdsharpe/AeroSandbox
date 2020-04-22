import casadi as cas
from aerosandbox.geometry import *


class OperatingPoint(AeroSandboxObject):
    def __init__(self,
                 density=1.225,  # kg/m^3
                 viscosity=1.81e-5,  # kg/m-s
                 velocity=10,  # m/s
                 mach=0,  # Freestream mach number
                 alpha=5,  # In degrees
                 beta=0,  # In degrees
                 p=0,  # About the body x-axis, in rad/sec
                 q=0,  # About the body y-axis, in rad/sec
                 r=0,  # About the body z-axis, in rad/sec
                 ):
        self.density = density
        self.viscosity = viscosity
        self.velocity = velocity
        self.mach = mach
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

    def compute_rotation_matrix_wind_to_geometry(self):
        # Computes the 3x3 rotation matrix required to go from wind axes to geometry axes.
        sinalpha = cas.sin(self.alpha * cas.pi / 180)
        cosalpha = cas.cos(self.alpha * cas.pi / 180)
        sinbeta = cas.sin(self.beta * cas.pi / 180)
        cosbeta = cas.cos(self.beta * cas.pi / 180)

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
        # Computes the effective velocity due to rotation at attrib_name set of points.
        # Input: attrib_name Nx3 array of points
        # Output: attrib_name Nx3 array of effective velocities
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

    def compute_reynolds(self, reference_length):
        """
        Computes a reynolds number with respect to a given reference length.
        :param reference_length: A reference length you choose [m]
        :return: Reynolds number [unitless]
        """
        return self.density * self.velocity * reference_length / self.viscosity


class AeroData: # TODO finish this class
    # A class where aerodynamic data is stored.
    # There is very little structure here. Suggested attributes you could add here:

    # # Forces / Moments
    # force_wind_axes,
    # force_geometry_axes,
    # CL,
    # CD,
    # CY,
    # Cl,
    # Cm,
    # Cn,

    # # Stability Derivatives

    def __init__(self,
                 force_wind_axes=None,
                 force_geometry_axes=None,
                 CL=None,
                 CD=None,
                 CY=None,
                 Cl=None,
                 Cm=None,
                 Cn=None,
                 stability_jacobian=None
                 ):
        self.force_wind_axes = force_wind_axes
        self.force_geometry_axes = force_geometry_axes
        self.CL = CL
        self.CD = CD
        self.CY = CY
        self.Cl = Cl
        self.Cm = Cm
        self.Cn = Cn
        self.stability_jacobian = stability_jacobian
