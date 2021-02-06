from numpy import *

from .array import array, length
from .arithmetic import sum, mean
from .calculus import diff, trapz
from .conditionals import where
import aerosandbox.numpy.linalg as linalg
from .logicals import clip
from .rotations import rotation_matrix_2D, rotation_matrix_3D
from .spacing import linspace, cosspace
from .surrogate_model_tools import smoothmax, sigmoid, blend
from .trig import degrees, radians, sind, cosd, tand, arcsind, arccosd, arctan2d
