import numpy as np
from numpy import pi


def sind(x):
    return np.sin(x * pi / 180)


def cosd(x):
    return np.cos(x * pi / 180)


def tand(x):
    return np.tan(x * pi / 180)


def arctan2d(y, x):
    return np.arctan2(y, x) * 180 / pi
