from aerosandbox.geometry import *
from aerosandbox import Atmosphere
import aerosandbox.numpy as np


class OperatingPoint(AeroSandboxObject):
    def __init__(self,
                 atmosphere: Atmosphere = Atmosphere(altitude=0),
                 velocity: float = 1.,
                 alpha: float = 0.,
                 beta: float = 0.,
                 p: float = 0.,
                 q: float = 0.,
                 r: float = 0.,
                 ):
        """
        An object that represents the instantaneous flight conditions of an aircraft.

        Args:
            atmosphere:
            velocity: The flight velocity, expressed in true airspeed. [m/s]
            alpha: The angle of attack. [degrees]
            beta: The sideslip angle. (Reminder: convention that a positive beta implies that the oncoming air comes from the pilot's right-hand side.) [degrees]
            p: The roll rate about the x_b axis. [rad/sec]
            q: The pitch rate about the y_b axis. [rad/sec]
            r: The yaw rate about the z_b axis. [rad/sec]
        """
        self.atmosphere = atmosphere
        self.velocity = velocity  # TODO rename "airspeed"?
        self.alpha = alpha
        self.beta = beta
        self.p = p
        self.q = q
        self.r = r

    def dynamic_pressure(self):
        """
        Dynamic pressure of the working fluid
        Returns:
            float: Dynamic pressure of the working fluid. [Pa]
        """
        return 0.5 * self.atmosphere.density() * self.velocity ** 2

    def total_pressure(self):
        """
        Total (stagnation) pressure of the working fluid.

        Assumes a calorically perfect gas (i.e. specific heats do not change across the isentropic deceleration).

        Note that `total pressure != static pressure + dynamic pressure`, due to compressibility effects.

        Returns: Total pressure of the working fluid. [Pa]

        """
        gamma = self.atmosphere.ratio_of_specific_heats()
        return self.atmosphere.pressure() * (
                1 + (gamma - 1) / 2 * self.mach() ** 2
        ) ** (
                       gamma / (gamma - 1)
               )

    def total_temperature(self):
        """
        Total (stagnation) temperature of the working fluid.

        Assumes a calorically perfect gas (i.e. specific heats do not change across the isentropic deceleration).

        Returns: Total temperature of the working fluid [K]

        """
        gamma = self.atmosphere.ratio_of_specific_heats()
        # return self.atmosphere.temperature() * (
        #         self.total_pressure() / self.atmosphere.pressure()
        # ) ** (
        #                (gamma - 1) / gamma
        #        )
        return self.atmosphere.temperature() * (
                1 + (gamma - 1) / 2 * self.mach() ** 2
        )

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
        Returns the Mach number associated with the current flight condition.
        """
        return self.velocity / self.atmosphere.speed_of_sound()

    def indicated_airspeed(self):
        """
        Returns the indicated airspeed associated with the current flight condition, in meters per second.
        """
        return np.sqrt(
            2 * (self.total_pressure() - self.atmosphere.pressure()) / self.atmosphere.density()
        )

    def equivalent_airspeed(self):
        """
        Returns the equivalent airspeed associated with the current flight condition, in meters per second.
        """
        return self.velocity * np.sqrt(
            self.atmosphere.density() / Atmosphere(altitude=0, method="isa").density()
        )

    def compute_rotation_matrix_wind_to_geometry(self) -> np.ndarray:
        """
        Computes the 3x3 rotation matrix that transforms from wind axes to geometry axes.

        Returns: a 3x3 rotation matrix.

        """

        alpha_rotation = np.rotation_matrix_3D(
            angle=np.radians(-self.alpha),
            axis=np.array([0, 1, 0]),
            _axis_already_normalized=True
        )
        beta_rotation = np.rotation_matrix_3D(
            angle=np.radians(-self.beta),
            axis=np.array([0, 0, 1]),
            _axis_already_normalized=True
        )
        axes_flip = np.rotation_matrix_3D(
            angle=np.pi,
            axis=np.array([0, 1, 0]),
            _axis_already_normalized=True
        )  # Since in geometry axes, X is downstream by convention, while in wind axes, X is upstream by convetion. Same with Z being up/down respectively.

        r = axes_flip @ alpha_rotation @ beta_rotation  # where "@" is the matrix multiplication operator

        return r

    def compute_freestream_direction_geometry_axes(self):
        # Computes the freestream direction (direction the wind is GOING TO) in the geometry axes
        return self.compute_rotation_matrix_wind_to_geometry() @ np.array([-1, 0, 0])

    def compute_freestream_velocity_geometry_axes(self):
        # Computes the freestream velocity vector (direction the wind is GOING TO) in geometry axes
        return self.compute_freestream_direction_geometry_axes() * self.velocity

    def compute_rotation_velocity_geometry_axes(self, points):
        # Computes the effective velocity-due-to-rotation at a set of points.
        # Input: a Nx3 array of points
        # Output: a Nx3 array of effective velocities
        angular_velocity_vector_geometry_axes = np.array([
            -self.p,
            self.q,
            -self.r
        ])  # signs convert from body axes to geometry axes

        a = angular_velocity_vector_geometry_axes
        b = points

        rotation_velocity_geometry_axes = np.transpose(np.array([
            a[1] * b[:, 2] - a[2] * b[:, 1],
            a[2] * b[:, 0] - a[0] * b[:, 2],
            a[0] * b[:, 1] - a[1] * b[:, 0]
        ]))

        rotation_velocity_geometry_axes = -rotation_velocity_geometry_axes  # negative sign, since we care about the velocity the WING SEES, not the velocity of the wing.

        return rotation_velocity_geometry_axes


if __name__ == '__main__':
    op_point = OperatingPoint()
