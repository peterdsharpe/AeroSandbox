import aerosandbox.numpy as np
from matplotlib import path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as mplPolygon
from aerosandbox.common import AeroSandboxObject


# TODO: Clean this up, casadi compat made a mess
class Polygon(AeroSandboxObject):
    def __init__(self, *args):
        
        if len(args) == 1:
            self._coordinates = args[0]  # Nx2 NumPy ndarray
        elif len(args) == 2:
            if args[0].shape[0] == 1 and args[1].shape[1] == 1:
                # Still try to combine them
                self._coordinates = np.hstack([*args])
            else:
                # Casadi only supports 2D matrices
                # TODO: This is odd casadi behavior
                self._x = args[0]
                self._y = args[1]
                
                # if type(args[0]) != np.ndarray:
                #     self._x = args[0].T
                # if type(args[1]) != np.ndarray:
                #     self._y = args[1].T
        
        # TODO: Make this better, fixes negative areas etc though
        # if self.area() < 0:
        #     self.coordinates = np.flip(coordinates, axis=0)

    @property
    def coordinates(self):
        if hasattr(self, "_coordinates"):
            return self._coordinates
        else:
            return (self.x(), self.y())
        
    @coordinates.setter
    def coordinates(self, value):
        self._coordinates = value

    def x(self):
        """
        Returns the x coordinates of the polygon. Equivalent to Polygon.coordinates[:,0].
        :return: X coordinates as a vector
        """
        if hasattr(self, "_coordinates"):
            return self._coordinates[:, 0]
        else:
            return self._x

    def y(self):
        """
        Returns the y coordinates of the polygon. Equivalent to Polygon.coordinates[:,1].
        :return: Y coordinates as a vector
        """
        if hasattr(self, "_coordinates"):
            return self._coordinates[:, 1]
        else:
            return self._y
    
    def _x_n(self):
        """
        Returns x coordinates offset by one.
        """
        if len(self.x().shape) > 1:
            return np.roll(self.x(), 1, axis=1)
        else:
            return np.roll(self.x(), 1, axis=0)
    
    def _y_n(self):
        """
        Returns y coordinates offset by one.
        """
        if len(self.y().shape) > 1:
            return np.roll(self.y(), 1, axis=1)
        else:
            return np.roll(self.y(), 1, axis=0)
        

    def n_points(self) -> int:
        """
        Returns the number of points/vertices/coordinates of the polygon.
        Analogous to len(coordinates)
        """
        try:
            return self.x().shape[0]
        except: pass
        
        try:
            return len(self.coordinates)
        except TypeError:
            try:
                return self.coordinates.shape[0]
            except AttributeError:
                return 0

    def contains_points(self,
                        x,
                        y,
                        ):
        assert type(self.coordinates) != tuple, 'Method does not work with symbolics.'
        
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
    
    @property
    def __a(self):
        x = self.x()
        y = self.y()
        x_n = self._x_n()  # x_next, or x_i+1
        y_n = self._y_n()  # y_next, or y_i+1

        a = x * y_n - x_n * y  # a is the area of the triangle bounded by a given point, the next point, and the origin.

        # Fix for 1D
        axis = 0
        if len(a.shape) == 2:
            axis = 1

        return a, axis        

    def area(self):
        # Returns the area of the polygon, in nondimensional (normalized to chord^2) units.
        a, axis = self.__a
        
        A = 0.5 * np.sum(a, axis=axis)  # area

        return A

    def centroid(self):
        # Returns the centroid of the polygon, in nondimensional (chord-normalized) units.
        x = self.x()
        y = self.y()
        x_n = self._x_n()  # x_next, or x_i+1
        y_n = self._y_n()  # y_next, or y_i+1

        a, axis = self.__a

        A = self.area()  # area

        x_c = 1 / (6 * A) * np.sum(a * (x + x_n), axis=axis)
        y_c = 1 / (6 * A) * np.sum(a * (y + y_n), axis=axis)
        centroid = np.array([x_c, y_c])

        return centroid

    def Ixx(self):
        # Returns the nondimensionalized Ixx moment of inertia, taken about the centroid.
        y = self.y()
        y_n = self._y_n()  # y_next, or y_i+1

        a, axis = self.__a

        A = self.area()
        centroid = self.centroid()

        Ixx = 1 / 12 * np.sum(a * (y ** 2 + y * y_n + y_n ** 2), axis=axis)

        Iuu = Ixx - A * centroid[1] ** 2

        return Iuu

    def Iyy(self):
        # Returns the nondimensionalized Iyy moment of inertia, taken about the centroid.
        x = self.x()
        x_n = self._x_n()  # x_next, or x_i+1

        a, axis = self.__a

        A = self.area()
        centroid = self.centroid()

        Iyy = 1 / 12 * np.sum(a * (x ** 2 + x * x_n + x_n ** 2), axis=axis)

        Ivv = Iyy - A * centroid[0] ** 2

        return Ivv

    def Ixy(self):
        # Returns the nondimensionalized product of inertia, taken about the centroid.
        x = self.x()
        y = self.y()
        x_n = self._x_n()  # x_next, or x_i+1
        y_n = self._y_n()  # y_next, or y_i+1

        a, axis = self.__a

        A = self.area()
        centroid = self.centroid()

        Ixy = 1 / 24 * np.sum(a * (x * y_n + 2 * x * y + 2 * x_n * y_n + x_n * y), axis=axis)

        Iuv = Ixy - A * centroid[0] * centroid[1]

        return Iuv

    def J(self):
        # Returns the nondimensionalized polar moment of inertia, taken about the centroid.
        J = self.Ixx() + self.Iyy()

        return J
    
    def max_x(self):
        """
        Finds the maximum x from the centroid
        """
        return np.max(self.x()-self.centroid()[0])
    
    def max_y(self):
        """
        Finds the maximum y from the centroid
        """
        return np.max(self.y()-self.centroid()[1])
    
    def max_xy(self):
        """
        Finds the maximum distance from the centroid
        """
        return np.max(np.sqrt(
            (self.x()-self.centroid()[0])**2 +
            (self.y()-self.centroid()[1])**2
            ))
    
    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.autoscale(enable=True) 
        
        x = self.x()            
        y = self.y()
        poly = mplPolygon( np.c_[x,y], facecolor='red', edgecolor='red', alpha=0.25)
        ax.add_patch(poly)
        
        plt.show()


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
