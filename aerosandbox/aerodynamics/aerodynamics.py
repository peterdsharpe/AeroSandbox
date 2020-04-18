from aerosandbox.geometry import *
from aerosandbox.performance import *


class AeroProblem(AeroSandboxObject):
    def __init__(self,
                 airplane,  # type: Airplane
                 op_point,  # type: OperatingPoint
                 ):
        self.airplane = airplane
        self.op_point = op_point
