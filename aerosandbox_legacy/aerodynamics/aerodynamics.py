import autograd.numpy as np
from autograd import grad
import scipy.linalg as sp_linalg
from numba import jit
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from ..plotting import *
from ..geometry import *
from ..performance import *

import cProfile
import functools
import os




class AeroProblem:
    def __init__(self,
                 airplane,  # Object of Airplane class
                 op_point,  # Object of OperatingPoint class
                 ):
        self.airplane = airplane
        self.op_point = op_point

def profile(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        try:
            profiler.enable()
            ret = func(*args, **kwargs)
            profiler.disable()
            return ret
        finally:
            filename = os.path.expanduser(
                os.path.join('~', func.__name__ + '.pstat')
            )
            profiler.dump_stats(filename)
            # profiler.print_stats()

    return wrapper