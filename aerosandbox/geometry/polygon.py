import aerosandbox.numpy as np
from matplotlib import path
from aerosandbox.common import AeroSandboxObject
from typing import Union

class Polygon(AeroSandboxObject):
    def __init__(self, coordinates):
        self.coordinates = coordinates  # Nx2 NumPy ndarray

    def x(self) -> np.ndarray:
        """
        Returns the x coordinates of the polygon. Equivalent to Polygon.coordinates[:,0].
        :return: X coordinates as a vector
        """
        return self.coordinates[:, 0]

    def y(self) -> np.ndarray:
        """
        Returns the y coordinates of the polygon. Equivalent to Polygon.coordinates[:,1].
        :return: Y coordinates as a vector
        """
        return self.coordinates[:, 1]

    def n_points(self) -> int:
        """
        Returns the number of points/vertices/coordinates of the polygon.
        Analogous to len(coordinates)
        """
        try:
            return len(self.coordinates)
        except TypeError:
            try:
                return self.coordinates.shape[0]
            except AttributeError:
                return 0

    def contains_points(self,
                        x: Union[float, np.ndarray],
                        y: Union[float, np.ndarray],
                        ) -> np.ndarray:
        """
        Returns a boolean array of whether or not some (x, y) point(s) are contained within the Polygon.

        Args:
            x: x-coordinate(s) of the query points.
            y: y-coordinate(s) of the query points.

        Returns: A boolean array of the same size as x and y.

        """
        x = np.array(x)
        y = np.array(y)
        try:
            input_shape = (x + y).shape
        except ValueError as e:  # If arrays are not broadcastable
            raise ValueError("Inputs x and y could not be broadcast together!") from e

        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

        points = np.hstack((x, y))

        contained = path.Path(
            vertices=self.coordinates
        ).contains_points(
            points
        )
        contained = np.array(contained).reshape(input_shape)

        return contained

    def area(self) -> float:
        """
        Returns the area of the polygon.
        """
        x = self.x()
        y = self.y()
        x_n = np.roll(x, -1)  # x_next, or x_i+1
        y_n = np.roll(y, -1)  # y_next, or y_i+1

        a = x * y_n - x_n * y  # a is the area of the triangle bounded by a given point, the next point, and the origin.

        A = 0.5 * np.sum(a)  # area

        return A

    def centroid(self) -> np.ndarray:
        """
        Returns the centroid of the polygon as a 1D np.ndarray of length 2.
        """
        x = self.x()
        y = self.y()
        x_n = np.roll(x, -1)  # x_next, or x_i+1
        y_n = np.roll(y, -1)  # y_next, or y_i+1

        a = x * y_n - x_n * y  # a is the area of the triangle bounded by a given point, the next point, and the origin.

        A = 0.5 * np.sum(a)  # area

        x_c = 1 / (6 * A) * np.sum(a * (x + x_n))
        y_c = 1 / (6 * A) * np.sum(a * (y + y_n))
        centroid = np.array([x_c, y_c])

        return centroid

    def Ixx(self):
        """
        Returns the nondimensionalized Ixx moment of inertia, taken about the centroid.
        """
        x = self.x()
        y = self.y()
        x_n = np.roll(x, -1)  # x_next, or x_i+1
        y_n = np.roll(y, -1)  # y_next, or y_i+1

        a = x * y_n - x_n * y  # a is the area of the triangle bounded by a given point, the next point, and the origin.

        A = 0.5 * np.sum(a)  # area

        x_c = 1 / (6 * A) * np.sum(a * (x + x_n))
        y_c = 1 / (6 * A) * np.sum(a * (y + y_n))
        centroid = np.array([x_c, y_c])

        Ixx = 1 / 12 * np.sum(a * (y ** 2 + y * y_n + y_n ** 2))

        Iuu = Ixx - A * centroid[1] ** 2

        return Iuu

    def Iyy(self):
        """
        Returns the nondimensionalized Iyy moment of inertia, taken about the centroid.
        """
        x = self.x()
        y = self.y()
        x_n = np.roll(x, -1)  # x_next, or x_i+1
        y_n = np.roll(y, -1)  # y_next, or y_i+1

        a = x * y_n - x_n * y  # a is the area of the triangle bounded by a given point, the next point, and the origin.

        A = 0.5 * np.sum(a)  # area

        x_c = 1 / (6 * A) * np.sum(a * (x + x_n))
        y_c = 1 / (6 * A) * np.sum(a * (y + y_n))
        centroid = np.array([x_c, y_c])

        Iyy = 1 / 12 * np.sum(a * (x ** 2 + x * x_n + x_n ** 2))

        Ivv = Iyy - A * centroid[0] ** 2

        return Ivv

    def Ixy(self):
        """
        Returns the nondimensionalized product of inertia, taken about the centroid.
        """
        x = self.x()
        y = self.y()
        x_n = np.roll(x, -1)  # x_next, or x_i+1
        y_n = np.roll(y, -1)  # y_next, or y_i+1

        a = x * y_n - x_n * y  # a is the area of the triangle bounded by a given point, the next point, and the origin.

        A = 0.5 * np.sum(a)  # area

        x_c = 1 / (6 * A) * np.sum(a * (x + x_n))
        y_c = 1 / (6 * A) * np.sum(a * (y + y_n))
        centroid = np.array([x_c, y_c])

        Ixy = 1 / 24 * np.sum(a * (x * y_n + 2 * x * y + 2 * x_n * y_n + x_n * y))

        Iuv = Ixy - A * centroid[0] * centroid[1]

        return Iuv

    def J(self):
        """
        Returns the nondimensionalized polar moment of inertia, taken about the centroid.
        """
        x = self.x()
        y = self.y()
        x_n = np.roll(x, -1)  # x_next, or x_i+1
        y_n = np.roll(y, -1)  # y_next, or y_i+1

        a = x * y_n - x_n * y  # a is the area of the triangle bounded by a given point, the next point, and the origin.

        A = 0.5 * np.sum(a)  # area

        x_c = 1 / (6 * A) * np.sum(a * (x + x_n))
        y_c = 1 / (6 * A) * np.sum(a * (y + y_n))
        centroid = np.array([x_c, y_c])

        Ixx = 1 / 12 * np.sum(a * (y ** 2 + y * y_n + y_n ** 2))

        Iyy = 1 / 12 * np.sum(a * (x ** 2 + x * x_n + x_n ** 2))

        J = Ixx + Iyy

        return J


def stack_coordinates(
        x: np.ndarray,
        y: np.ndarray
) -> np.ndarray:
    """
    Stacks a pair of x, y coordinate arrays into a Nx2 ndarray.
    Args:
        x: A 1D ndarray of x-coordinates
        y: A 1D ndarray of y-coordinates

    Returns: A Nx2 ndarray of [x, y] coordinates.

    """
    return np.vstack((x, y)).T
