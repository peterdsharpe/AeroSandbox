from aerosandbox.geometry import *
from aerosandbox.performance import *


class AeroProblem(AeroSandboxObject):
    def __init__(self,
                 airplane: Airplane,
                 op_point: OperatingPoint,
                 ):
        self.airplane = airplane
        self.op_point = op_point
