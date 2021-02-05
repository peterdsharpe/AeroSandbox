import numpy as np
from numpy import pi


def degrees(x):
    """Converts an input x from radians to degrees"""
    return x * 180 / pi


def radians(x):
    """Converts an input x from degrees to radians"""
    return x * pi / 180


def sind(x):
    """Returns the sin of an angle x, given in degrees"""
    return np.sin(radians(x))


def cosd(x):
    """Returns the cos of an angle x, given in degrees"""
    return np.cos(radians(x))


def tand(x):
    """Returns the tangent of an angle x, given in degrees"""
    return np.tan(radians(x))


def arctan2d(y, x):
    """Returns the angle associated with arctan(y, x), in degrees"""
    return deg(np.arctan2(y, x))
