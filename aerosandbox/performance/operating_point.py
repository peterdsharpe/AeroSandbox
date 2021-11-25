from aerosandbox.geometry import *
from aerosandbox import Atmosphere
import aerosandbox.numpy as np
from typing import Tuple, Union


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

    def convert_axes(self,
                     x_from: Union[float, np.ndarray],
                     y_from: Union[float, np.ndarray],
                     z_from: Union[float, np.ndarray],
                     from_axes: str,
                     to_axes: str,
                     ) -> Tuple[float, float, float]:
        """
        Converts a vector [x_from, y_from, z_from], as given in the `from_axes` frame, to an equivalent vector [x_to,
        y_to, z_to], as given in the `to_axes` frame.

        Both `from_axes` and `to_axes` should be a string, one of:
            * "geometry"
            * "body"
            * "wind"
            * "stability"

        This whole function is vectorized, both over the vector and the OperatingPoint (e.g., a vector of
        `OperatingPoint.alpha` values)

        Wind axes rotations are taken from Eq. 6.7 in Sect. 6.2.2 of Drela's Flight Vehicle Aerodynamics textbook,
        with axes corrections to go from [D, Y, L] to true wind axes (and same for geometry to body axes).

        Args:
            x_from: x-component of the vector, in `from_axes` frame.
            y_from: y-component of the vector, in `from_axes` frame.
            z_from: z-component of the vector, in `from_axes` frame.
            from_axes: The axes to convert from.
            to_axes: The axes to convert to.

        Returns: The x-, y-, and z-components of the vector, in `to_axes` frame. Given as a tuple.

        """
        if from_axes == "geometry":
            x_b = -x_from
            y_b = y_from
            z_b = -z_from
        elif from_axes == "body":
            x_b = x_from
            y_b = y_from
            z_b = z_from
        elif from_axes == "wind":
            sa = np.sind(self.alpha)
            ca = np.cosd(self.alpha)
            sb = np.sind(self.beta)
            cb = np.cosd(self.beta)
            x_b = (cb * ca) * x_from + (-sb * ca) * y_from + (-sa) * z_from
            y_b = (sb) * x_from + (cb) * y_from  # Note: z term is 0; not forgotten.
            z_b = (cb * sa) * x_from + (-sb * sa) * y_from + (ca) * z_from
        elif from_axes == "stability":
            sa = np.sind(self.alpha)
            ca = np.cosd(self.alpha)
            x_b = ca * x_from - sa * z_from
            y_b = y_from
            z_b = sa * x_from + ca * z_from
        else:
            raise ValueError("Bad value of `from_axes`!")

        if to_axes == "geometry":
            x_to = -x_b
            y_to = y_b
            z_to = -z_b
        elif to_axes == "body":
            x_to = x_b
            y_to = y_b
            z_to = z_b
        elif to_axes == "wind":
            sa = np.sind(self.alpha)
            ca = np.cosd(self.alpha)
            sb = np.sind(self.beta)
            cb = np.cosd(self.beta)
            x_to = (cb * ca) * x_b + (sb) * y_b + (cb * sa) * z_b
            y_to = (-sb * ca) * x_b + (cb) * y_b + (-sb * sa) * z_b
            z_to = (-sa) * x_b + (ca) * z_b  # Note: y term is 0; not forgotten.
        elif to_axes == "stability":
            sa = np.sind(self.alpha)
            ca = np.cosd(self.alpha)
            x_to = ca * x_b + sa * z_b
            y_to = y_b
            z_to = -sa * x_b + ca * z_b
        else:
            raise ValueError("Bad value of `to_axes`!")

        return x_to, y_to, z_to

    def compute_rotation_matrix_wind_to_geometry(self) -> np.ndarray:
        """
        Computes the 3x3 rotation matrix that transforms from wind axes to geometry axes.

        Returns: a 3x3 rotation matrix.

        """

        alpha_rotation = np.rotation_matrix_3D(
            angle=np.radians(-self.alpha),
            axis="y",
        )
        beta_rotation = np.rotation_matrix_3D(
            angle=np.radians(-self.beta),
            axis="z",
        )
        axes_flip = np.rotation_matrix_3D(
            angle=np.pi,
            axis="y",
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
