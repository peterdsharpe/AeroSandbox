import numpy as np
from abc import ABC, abstractmethod


class Singularity(ABC):
    @abstractmethod
    def get_potential_at(self, points: np.ndarray) -> np.ndarray:
        """
        Returns the velocity potential at a given set of points.

        The velocity vector is the gradient of the velocity potential.

        Args:

            points: A numpy array of shape (N, 2) where N is the number of points at which to evaluate the velocity
                potential.

        Returns: A numpy array of shape (N), giving the velocity potential at each point.
        """
        pass

    @abstractmethod
    def get_streamfunction_at(self, points: np.ndarray) -> np.ndarray:
        """
        Returns the streamfunction at a given set of points.

        The streamfunction is a function where isolines are streamlines of the flow.

        The gradient of the streamfunction is always exactly 90 degrees counterclockwise from the gradient of the
            velocity potential function.

        Args:

            points: A numpy array of shape (N, 2) where N is the number of points at which to evaluate the streamfunction.

        Returns: A numpy array of shape (N), giving the streamfunction at each point.
        """
        pass

    @abstractmethod
    def get_x_velocity_at(self, points: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_y_velocity_at(self, points: np.ndarray) -> np.ndarray:
        pass


class Freestream(Singularity):
    def __init__(
        self,
        u,
        v,
    ):
        self.u = u
        self.v = v

    def get_potential_at(self, points: np.ndarray):
        return self.u * points[:, 0] + self.v * points[:, 1]

    def get_streamfunction_at(self, points: np.ndarray):
        return -self.v * points[:, 0] + self.u * points[:, 1]

    def get_x_velocity_at(self, points: np.ndarray):
        return self.u * np.ones_like(points[:, 0])

    def get_y_velocity_at(self, points: np.ndarray):
        return self.v * np.ones_like(points[:, 0])


class Source(Singularity):
    def __init__(
        self,
        strength,
        x,  # x-location
        y,  # y-location
    ):
        self.strength = strength
        self.x = x
        self.y = y

    def get_potential_at(self, points: np.ndarray):
        return (
            self.strength
            / (2 * np.pi)
            * np.log(
                np.sqrt((points[:, 0] - self.x) ** 2 + (points[:, 1] - self.y) ** 2)
            )
        )

    def get_streamfunction_at(self, points: np.ndarray):
        return (
            self.strength
            / (2 * np.pi)
            * np.arctan2(points[:, 1] - self.y, points[:, 0] - self.x)
        )

    def get_x_velocity_at(self, points: np.ndarray):
        return (
            self.strength
            / (2 * np.pi)
            * (points[:, 0] - self.x)
            / ((points[:, 0] - self.x) ** 2 + (points[:, 1] - self.y) ** 2)
        )

    def get_y_velocity_at(self, points: np.ndarray):
        return (
            self.strength
            / (2 * np.pi)
            * (points[:, 1] - self.y)
            / ((points[:, 0] - self.x) ** 2 + (points[:, 1] - self.y) ** 2)
        )


class Vortex(Singularity):
    def __init__(
        self,
        strength,
        x,  # x-location
        y,  # y-location
    ):
        self.strength = strength
        self.x = x
        self.y = y

    def get_potential_at(self, points: np.ndarray):
        return (
            self.strength
            / (2 * np.pi)
            * np.arctan2(points[:, 1] - self.y, points[:, 0] - self.x)
        )

    def get_streamfunction_at(self, points: np.ndarray):
        return (
            -self.strength
            / (2 * np.pi)
            * np.log(
                np.sqrt((points[:, 0] - self.x) ** 2 + (points[:, 1] - self.y) ** 2)
            )
        )

    def get_x_velocity_at(self, points: np.ndarray):
        return (
            self.strength
            / (2 * np.pi)
            * -(points[:, 1] - self.y)
            / ((points[:, 0] - self.x) ** 2 + (points[:, 1] - self.y) ** 2)
        )

    def get_y_velocity_at(self, points: np.ndarray):
        return (
            self.strength
            / (2 * np.pi)
            * (points[:, 0] - self.x)
            / ((points[:, 0] - self.x) ** 2 + (points[:, 1] - self.y) ** 2)
        )


class Doublet(Singularity):
    """
    Note that there are differing sign conventions for doublet strength in the literature.

    Consider the case of a unit-strength doublet, at the origin (0, 0), facing "to the right" (alpha = 0).

    We use the "MIT convention", where a positive strength induces a flow to right along the doublet axis (the "core"). So,
    points on the x-axis itself will have an induced velocity in the +x direction. Points far from the doublet axis (on "the donut") will in general
    have induced velocity in the -x direction. See a diagram of the MIT convention:
        > https://web.mit.edu/fluids-modules/www/potential_flows/LecturesHTML/lec1011/node24.html

    The alternative is the "Cambridge convention", where this is flipped:
        > http://www-mdp.eng.cam.ac.uk/web/library/enginfo/aerothermal_dvd_only/aero/fprops/poten/node33.html

    We use the MIT convention in this code implementation, but either convention is fine as long as you're consistent.
    """

    def __init__(
        self,
        strength,
        x,  # x-location
        y,  # y-location
        alpha,  # angle, in radians
    ):
        self.strength = strength
        self.x = x
        self.y = y
        self.alpha = alpha

    def get_potential_at(self, points: np.ndarray):
        return (
            -self.strength
            / (2 * np.pi)
            * (
                (points[:, 0] - self.x) * np.cos(self.alpha)
                + (points[:, 1] - self.y) * np.sin(self.alpha)
            )
            / ((points[:, 0] - self.x) ** 2 + (points[:, 1] - self.y) ** 2)
        )

    def get_streamfunction_at(self, points: np.ndarray):
        return (
            self.strength
            / (2 * np.pi)
            * (
                (points[:, 0] - self.x) * np.sin(self.alpha)
                + (points[:, 1] - self.y) * np.cos(self.alpha)
            )
            / ((points[:, 0] - self.x) ** 2 + (points[:, 1] - self.y) ** 2)
        )

    def get_x_velocity_at(self, points: np.ndarray):
        return (
            -self.strength
            / (2 * np.pi)
            * (
                ((points[:, 0] - self.x) ** 2 + (points[:, 1] - self.y) ** 2)
                * np.cos(self.alpha)
                - 2
                * (points[:, 0] - self.x)
                * (
                    (points[:, 0] - self.x) * np.cos(self.alpha)
                    + (points[:, 1] - self.y) * np.sin(self.alpha)
                )
            )
            / ((points[:, 0] - self.x) ** 2 + (points[:, 1] - self.y) ** 2) ** 2
        )

    def get_y_velocity_at(self, points: np.ndarray):
        return (
            -self.strength
            / (2 * np.pi)
            * (
                ((points[:, 0] - self.x) ** 2 + (points[:, 1] - self.y) ** 2)
                * np.sin(self.alpha)
                - 2
                * (points[:, 1] - self.y)
                * (
                    (points[:, 0] - self.x) * np.cos(self.alpha)
                    + (points[:, 1] - self.y) * np.sin(self.alpha)
                )
            )
            / ((points[:, 0] - self.x) ** 2 + (points[:, 1] - self.y) ** 2) ** 2
        )


class LineSource(Singularity):
    def __init__(
        self,
        strength,
        x1,  # x-location of start
        y1,  # y-location of start
        x2,  # x-location of end
        y2,  # y-location of end
    ):
        self.strength = strength
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def get_potential_at(self, points: np.ndarray):
        A = np.array(
            [
                [self.x2 - self.x1, self.y2 - self.y1],
                [self.y1 - self.y2, self.x2 - self.x1],
            ]
        ) / ((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2)
        b = np.array([self.x1, self.y1])

        points_transformed = np.transpose(A @ (points - b).T)
        xf = points_transformed[:, 0]
        yf = points_transformed[:, 1]

        potential = (
            self.strength
            / (2 * np.pi)
            * (
                yf * (np.arctan(xf / yf) - np.arctan((xf - 1) / yf))
                + xf * np.log(xf**2 + yf**2) / 2
                - np.log(np.sqrt((xf - 1) ** 2 + yf**2)) * (xf - 1)
                - 1
            )
        )

        return potential

    def get_streamfunction_at(self, points: np.ndarray):
        A = np.array(
            [
                [self.x2 - self.x1, self.y2 - self.y1],
                [self.y1 - self.y2, self.x2 - self.x1],
            ]
        ) / ((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2)
        b = np.array([self.x1, self.y1])

        points_transformed = np.transpose(A @ (points - b).T)
        xf = points_transformed[:, 0]
        yf = points_transformed[:, 1]

        s1 = -yf / 2 + xf / 2 * 1j
        s2 = yf / 2 + xf / 2 * 1j
        s3 = xf - 1 + yf * 1j

        streamfunction = (
            self.strength
            / (2 * np.pi)
            * (
                -np.log(s3 / np.sqrt((xf - 1) ** 2 + yf**2)) * 1j
                - np.log(xf + yf * 1j) * s1
                + np.log(s3) * s1
                + np.log(-xf + yf * 1j) * s2
                - np.log(1 - xf + yf * 1j) * s2
            )
        )
        streamfunction = np.real(streamfunction)

        return streamfunction

    def get_x_velocity_at(self, points: np.ndarray):
        A = np.array(
            [
                [self.x2 - self.x1, self.y2 - self.y1],
                [self.y1 - self.y2, self.x2 - self.x1],
            ]
        ) / ((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2)
        b = np.array([self.x1, self.y1])

        points_transformed = np.transpose(A @ (points - b).T)
        xf = points_transformed[:, 0]
        yf = points_transformed[:, 1]

        x_vel = (
            -self.strength
            / (4 * np.pi)
            * (
                np.log(xf + yf * 1j) * 1j
                - np.log(xf - 1 + yf * 1j) * 1j
                - np.log(-xf + yf * 1j) * 1j
                + np.log(1 - xf + yf * 1j) * 1j
            )
        )

        x_vel = np.real(x_vel)

        scalefactor = np.sqrt((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2)
        x_vel /= scalefactor

        return x_vel

    def get_y_velocity_at(self, points: np.ndarray):
        A = np.array(
            [
                [self.x2 - self.x1, self.y2 - self.y1],
                [self.y1 - self.y2, self.x2 - self.x1],
            ]
        ) / ((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2)
        b = np.array([self.x1, self.y1])

        points_transformed = np.transpose(A @ (points - b).T)
        xf = points_transformed[:, 0]
        yf = points_transformed[:, 1]

        y_vel = (
            self.strength
            / (4 * np.pi)
            * (
                np.log(xf + yf * 1j)
                - np.log(xf - 1 + yf * 1j)
                + np.log(-xf + yf * 1j)
                - np.log(1 - xf + yf * 1j)
            )
        )

        y_vel = np.real(y_vel)

        scalefactor = np.sqrt((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2)
        y_vel /= scalefactor

        return y_vel


class LineVortex(Singularity):
    def __init__(
        self,
        strength,
        x1,  # x-location of start
        y1,  # y-location of start
        x2,  # x-location of end
        y2,  # y-location of end
    ):
        self.strength = strength
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def get_potential_at(self, points: np.ndarray):
        A = np.array(
            [
                [self.x2 - self.x1, self.y2 - self.y1],
                [self.y1 - self.y2, self.x2 - self.x1],
            ]
        ) / ((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2)
        b = np.array([self.x1, self.y1])

        points_transformed = np.transpose(A @ (points - b).T)
        xf = points_transformed[:, 0]
        yf = points_transformed[:, 1]

        s1 = -yf / 2 + xf / 2 * 1j
        s2 = yf / 2 + xf / 2 * 1j
        s3 = xf - 1 + yf * 1j

        potential = (
            self.strength
            / (2 * np.pi)
            * (
                -np.log(s3 / np.sqrt((xf - 1) ** 2 + yf**2)) * 1j
                - np.log(xf + yf * 1j) * s1
                + np.log(s3) * s1
                + np.log(-xf + yf * 1j) * s2
                - np.log(1 - xf + yf * 1j) * s2
            )
        )
        potential = np.real(potential)

        return potential

    def get_streamfunction_at(self, points: np.ndarray):
        A = np.array(
            [
                [self.x2 - self.x1, self.y2 - self.y1],
                [self.y1 - self.y2, self.x2 - self.x1],
            ]
        ) / ((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2)
        b = np.array([self.x1, self.y1])

        points_transformed = np.transpose(A @ (points - b).T)
        xf = points_transformed[:, 0]
        yf = points_transformed[:, 1]

        streamfunction = (
            -self.strength
            / (2 * np.pi)
            * (
                yf * (np.arctan(xf / yf) - np.arctan((xf - 1) / yf))
                + xf * np.log(xf**2 + yf**2) / 2
                - np.log(np.sqrt((xf - 1) ** 2 + yf**2)) * (xf - 1)
                - 1
            )
        )

        return streamfunction

    def get_x_velocity_at(self, points: np.ndarray):
        A = np.array(
            [
                [self.x2 - self.x1, self.y2 - self.y1],
                [self.y1 - self.y2, self.x2 - self.x1],
            ]
        ) / ((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2)
        b = np.array([self.x1, self.y1])

        points_transformed = np.transpose(A @ (points - b).T)
        xf = points_transformed[:, 0]
        yf = points_transformed[:, 1]

        x_vel = (
            self.strength
            / (4 * np.pi)
            * (np.log(xf**2 - 2 * xf + yf**2 + 1) - np.log(xf**2 + yf**2))
        )
        scalefactor = np.sqrt((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2)
        x_vel /= scalefactor

        return x_vel

    def get_y_velocity_at(self, points: np.ndarray):
        A = np.array(
            [
                [self.x2 - self.x1, self.y2 - self.y1],
                [self.y1 - self.y2, self.x2 - self.x1],
            ]
        ) / ((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2)
        b = np.array([self.x1, self.y1])

        points_transformed = np.transpose(A @ (points - b).T)
        xf = points_transformed[:, 0]
        yf = points_transformed[:, 1]

        y_vel = (
            -self.strength
            / (2 * np.pi)
            * (np.arctan(xf / yf) - np.arctan((xf - 1) / yf))
        )
        scalefactor = np.sqrt((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2)
        y_vel /= scalefactor

        return y_vel
