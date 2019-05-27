import numpy as np
import math
import matplotlib.pyplot as plt
from .Plotting import *


class OperatingPoint:
    def __init__(self,
                 velocity=10,
                 alpha=5,
                 beta=0,
                 p=0,
                 q=0,
                 r=0,
                 ):
        self.velocity = velocity
        self.alpha = alpha
        self.beta = beta
        self.p = p
        self.q = q
        self.r = r

    def compute_freestream_velocity(self):
        # Computes the freestream velocity vector in aircraft geometry coordinates

        sinalpha = np.sin(np.radians(self.alpha))
        cosalpha = np.cos(np.radians(self.alpha))
        sinbeta = np.sin(np.radians(self.beta))
        cosbeta = np.cos(np.radians(self.beta))

        vel_vec = self.velocity * np.array([
            cosalpha * cosbeta,
            -sinbeta,
            sinalpha * cosbeta,
        ])

        return vel_vec
