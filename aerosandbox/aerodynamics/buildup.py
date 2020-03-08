from .aerodynamics import *
from ..geometry import *

class AeroBuildup(AeroProblem):
    def __init__(self,
                 airplane,  # type: Airplane
                 op_point,  # type: op_point
                 ):
        super().__init__(airplane, op_point)

