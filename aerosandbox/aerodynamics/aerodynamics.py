from ..geometry import *
from ..performance import *

class AeroProblem:
    def __init__(self,
                 airplane, # type: Airplane
                 op_point, # type: OperatingPoint
                 ):
        self.airplane = airplane
        self.op_point = op_point

    def substitute_solution(self, sol):
        """
        Substitutes a solution from CasADi's solver.
        :param sol:
        :return:
        """
        for attrib_name in dir(self):
            attrib_orig = getattr(self, attrib_name)
            if isinstance(attrib_orig, bool) or isinstance(attrib_orig, int):
                continue
            try:
                setattr(self, attrib_name, sol.value(attrib_orig))
            except NotImplementedError:
                pass
            if isinstance(attrib_orig, list):
                try:
                    new_attrib_orig = []
                    for item in attrib_orig:
                        new_attrib_orig.append(item.substitute_solution(sol))
                    setattr(self, attrib_name, new_attrib_orig)
                except:
                    pass
            if isinstance(attrib_orig, Airplane):
                setattr(self, attrib_name, attrib_orig.substitute_solution(sol))
            if isinstance(attrib_orig, OperatingPoint):
                setattr(self, attrib_name, attrib_orig.substitute_solution(sol))
        return self