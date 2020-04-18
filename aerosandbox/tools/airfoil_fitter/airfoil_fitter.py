"""
Functions to fit automatic-differentiable models to aerodynamic data from an airfoil.
Requires the xfoil package from PyPI
"""
from aerosandbox.geometry import *


class AirfoilFitter():
    def __init__(self,
                 airfoil,  # type: Airfoil
                 ):
        self.airfoil = airfoil


af = AirfoilFitter(Airfoil("e216"))

if __name__ == '__main__':
    pass