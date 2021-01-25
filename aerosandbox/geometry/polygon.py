import numpy as np

class Polygon():
    def __init__(self, coordinates):
        self.coordinates = coordinates  # Nx2 NumPy ndarray

    def x(self):
        """
        Returns the x coordinates of the polygon. Equivalent to Polygon.coordinates[:,0].
        :return: X coordinates as a vector
        """
        return self.coordinates[:, 0]

    def y(self):
        """
        Returns the y coordinates of the polygon. Equivalent to Polygon.coordinates[:,1].
        :return: Y coordinates as a vector
        """
        return self.coordinates[:, 1]

    def TE_thickness(self):
        # Returns the thickness of the trailing edge of the polygon, in nondimensional (chord-normalized) units.
        return self.local_thickness(x_over_c=1)

    def TE_angle(self):
        # Returns the trailing edge angle of the polygon, in degrees
        upper_TE_vec = self.coordinates[0, :] - self.coordinates[1, :]
        lower_TE_vec = self.coordinates[-1, :] - self.coordinates[-2, :]

        return 180 / np.pi * (np.arctan2(
            upper_TE_vec[0] * lower_TE_vec[1] - upper_TE_vec[1] * lower_TE_vec[0],
            upper_TE_vec[0] * lower_TE_vec[0] + upper_TE_vec[1] * upper_TE_vec[1]
        ))

    def area(self):
        # Returns the area of the polygon, in nondimensional (normalized to chord^2) units.
        x = self.x()
        y = self.y()
        x_n = np.roll(x, -1)  # x_next, or x_i+1
        y_n = np.roll(y, -1)  # y_next, or y_i+1

        a = x * y_n - x_n * y  # a is the area of the triangle bounded by a given point, the next point, and the origin.

        A = 0.5 * np.sum(a)  # area

        return A

    def centroid(self):
        # Returns the centroid of the polygon, in nondimensional (chord-normalized) units.
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
        # Returns the nondimensionalized Ixx moment of inertia, taken about the centroid.
        x = self.x()
        y = self.y()
        x_n = np.roll(x, -1)  # x_next, or x_i+1
        y_n = np.roll(y, -1)  # y_next, or y_i+1

        a = x * y_n - x_n * y  # a is the area of the triangle bounded by a given point, the next point, and the origin.

        A = 0.5 * np.sum(a)  # area

        x_c = 1 / (6 * A) * cas.sum1(a * (x + x_n))
        y_c = 1 / (6 * A) * cas.sum1(a * (y + y_n))
        centroid = np.array([x_c, y_c])

        Ixx = 1 / 12 * np.sum(a * (y ** 2 + y * y_n + y_n ** 2))

        Iuu = Ixx - A * centroid[1] ** 2

        return Iuu

    def Iyy(self):
        # Returns the nondimensionalized Iyy moment of inertia, taken about the centroid.
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
        # Returns the nondimensionalized product of inertia, taken about the centroid.
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
        # Returns the nondimensionalized polar moment of inertia, taken about the centroid.
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
