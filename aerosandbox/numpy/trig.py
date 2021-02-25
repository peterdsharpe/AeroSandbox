import numpy as _onp
from numpy import pi as _pi


def degrees(x):
    """Converts an input x from radians to degrees"""
    return x * 180 / _pi


def radians(x):
    """Converts an input x from degrees to radians"""
    return x * _pi / 180


def sind(x):
    """Returns the sin of an angle x, given in degrees"""
    return _onp.sin(radians(x))


def cosd(x):
    """Returns the cos of an angle x, given in degrees"""
    return _onp.cos(radians(x))


def tand(x):
    """Returns the tangent of an angle x, given in degrees"""
    return _onp.tan(radians(x))


def arcsind(x):
    """Returns the arcsin of an x, in degrees"""
    return degrees(_onp.arcsin(x))


def arccosd(x):
    """Returns the arccos of an x, in degrees"""
    return degrees(_onp.arccos(x))


def arctan2d(y, x):
    """Returns the angle associated with arctan(y, x), in degrees"""
    return degrees(_onp.arctan2(y, x))
