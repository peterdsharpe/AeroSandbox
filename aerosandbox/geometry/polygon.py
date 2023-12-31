import aerosandbox.numpy as np
from matplotlib import path
from aerosandbox.common import AeroSandboxObject
from typing import Union


class Polygon(AeroSandboxObject):
    def __init__(self,
                 coordinates: np.ndarray
                 ):
        """
        Creates a polygon object.

        Args:
            coordinates: An Nx2 NumPy ndarray of [x, y] coordinates for the polygon.
        """
        self.coordinates = np.array(coordinates)

    def __repr__(self):
        return f"Polygon ({self.n_points()} points)"

    def __eq__(self, other):
        return np.all(self.coordinates == other.coordinates)

    def __ne__(self, other):
        return not self.__eq__(other)

    def x(self) -> np.ndarray:
        """
        Returns the x coordinates of the polygon. Equivalent to Polygon.coordinates[:,0].

        Returns:
            X coordinates as a vector
        """
        return self.coordinates[:, 0]

    def y(self) -> np.ndarray:
        """
        Returns the y coordinates of the polygon. Equivalent to Polygon.coordinates[:,1].

        Returns:
            Y coordinates as a vector
        """
        return self.coordinates[:, 1]

    def n_points(self) -> int:
        """
        Returns the number of points/vertices/coordinates of the polygon.
        """
        try:
            return len(self.coordinates)
        except TypeError:
            try:
                return self.coordinates.shape[0]
            except AttributeError:
                return 0

    def scale(self,
              scale_x: float = 1.,
              scale_y: float = 1.,
              ) -> 'Polygon':
        """
        Scales a Polygon about the origin.
        Args:
            scale_x: Amount to scale in the x-direction.
            scale_y: Amount to scale in the y-direction.

        Returns: The scaled Polygon.
        """
        x = self.x() * scale_x
        y = self.y() * scale_y

        return Polygon(
            coordinates=np.stack((x, y), axis=1)
        )

    def translate(self,
                  translate_x: float = 0.,
                  translate_y: float = 0.,
                  ) -> 'Polygon':
        """
        Translates a Polygon by a given amount.
        Args:
            translate_x: Amount to translate in the x-direction
            translate_y: Amount to translate in the y-direction

        Returns: The translated Polygon.

        """
        x = self.x() + translate_x
        y = self.y() + translate_y

        return Polygon(
            coordinates=np.stack((x, y), axis=1)
        )

    def rotate(self,
               angle: float,
               x_center: float = 0.,
               y_center: float = 0.
               ) -> 'Polygon':
        """
        Rotates a Polygon clockwise by the specified amount, in radians.

        Rotates about the point (x_center, y_center), which is (0, 0) by default.

        Args:
            angle: Angle to rotate, counterclockwise, in radians.

            x_center: The x-coordinate of the center of rotation.

            y_center: The y-coordinate of the center of rotation.

        Returns: The rotated Polygon.

        """
        ### Translate
        translation = np.reshape(np.array([x_center, y_center]), (1, 2))
        coordinates = self.coordinates - translation

        ### Rotate
        rotation_matrix = np.rotation_matrix_2D(
            angle=angle,
        )
        coordinates = (rotation_matrix @ coordinates.T).T

        ### Translate
        coordinates = coordinates + translation

        return Polygon(
            coordinates=coordinates
        )

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

    def perimeter(self) -> float:
        """
        Returns the perimeter of the polygon.
        """
        dx = np.diff(self.x())
        dy = np.diff(self.y())
        ds = (
                     dx ** 2 +
                     dy ** 2
             ) ** 0.5

        return np.sum(ds)

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

    def write_sldcrv(self,
                     filepath: str = None
                     ):
        """
        Writes a .sldcrv (SolidWorks curve) file corresponding to this Polygon to a filepath.

        Args:
            filepath: A filepath (including the filename and .sldcrv extension) [string]
                if None, this function returns the .sldcrv file as a string.

        Returns: None

        """
        string = "\n".join(
            [
                "%f %f 0" % tuple(coordinate)
                for coordinate in self.coordinates
            ]
        )

        if filepath is not None:
            with open(filepath, "w+") as f:
                f.write(string)

        return string

    def contains_points(self,
                        x: Union[float, np.ndarray],
                        y: Union[float, np.ndarray],
                        ) -> Union[float, np.ndarray]:
        """
        Returns a boolean array of whether some (x, y) point(s) are contained within the Polygon.

        Note: This function is unfortunately not automatic-differentiable.

        Args:
            x: x-coordinate(s) of the query points.
            y: y-coordinate(s) of the query points.

        Returns:

            A boolean array of the same size as x and y, with values corresponding to whether the points are
            inside the Polygon.

        """
        x = np.array(x)
        y = np.array(y)
        try:
            input_shape = (x + y).shape
        except ValueError as e:  # If arrays are not broadcastable
            raise ValueError("Inputs x and y could not be broadcast together!") from e

        x = np.reshape(x, (-1, 1))
        y = np.reshape(y, (-1, 1))

        points = np.hstack((x, y))

        contained = path.Path(
            vertices=self.coordinates
        ).contains_points(
            points
        )
        contained = np.array(contained).reshape(input_shape)

        return contained

    def as_shapely_polygon(self):
        """
        Returns a Shapely Polygon object representing this polygon.

        Shapely is a Python library for 2D geometry operations. While it is more powerful than this class (e.g.,
            allows for union/intersection calculation between Polygons), it is not automatic-differentiable.
        """
        import shapely
        return shapely.Polygon(self.coordinates)

    def jaccard_similarity(self,
                           other: "Polygon"
                           ):
        """
        Calculates the Jaccard similarity between this polygon and another polygon.

        Note: This function is unfortunately not automatic-differentiable.

        Args:
            other: The other polygon to compare to.

        Returns:
            The Jaccard similarity between this polygon and the other polygon.
                * 0 if the polygons are completely disjoint
                * 1 if the polygons are identical

        """
        p1 = self.as_shapely_polygon()
        p2 = other.as_shapely_polygon()
        intersection = p1.intersection(p2).area
        union = p1.area + p2.area - intersection
        similarity = intersection / union if union != 0 else 0
        return similarity

    def draw(self,
             set_equal=True,
             color=None,
             **kwargs
             ):
        """
        Draws the Polygon on the current matplotlib axis.

        Args:

            set_equal: Whether to set the aspect ratio of the plot to be equal.

            **kwargs: Keyword arguments to pass to the matplotlib.pyplot.fill function.
                See: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.fill.html

        Returns: None (draws on the current matplotlib axis)

        """
        import matplotlib.pyplot as plt

        if color is None:
            color = plt.gca()._get_lines.get_next_color()

        plt.fill(
            self.x(),
            self.y(),
            color=color,
            alpha=0.5,
            **kwargs
        )

        if set_equal:
            plt.gca().set_aspect("equal", adjustable='box')


if __name__ == '__main__':

    theta = np.linspace(0, 2 * np.pi, 1000)
    r = np.sin(theta) * np.sqrt(np.abs(np.cos(theta))) / (np.sin(theta) + 7 / 5) - 2 * np.sin(theta) + 2
    heart = Polygon(np.stack((r * np.cos(theta), r * np.sin(theta)), axis=1))

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    heart.draw()
    heart.scale(0.7, 0.7).translate(2, 1).rotate(np.radians(15)).draw()
    plt.show()
