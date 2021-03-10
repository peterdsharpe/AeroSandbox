from aerosandbox import AeroSandboxObject
import aerosandbox.numpy as np
import abc

class AirfoilAerodynamicsFunction(AeroSandboxObject):
    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self, alpha, Re, mach, deflection):
        return Cl

