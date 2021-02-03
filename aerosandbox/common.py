from aerosandbox.optimization import *


class AeroSandboxObject:
    def substitute_solution(self, sol):
        """
        Substitutes a solution from CasADi's solver.

        In-place operation. To make it not in-place, do a copy.deepcopy(x) first.
        :param sol: OptiSol object.
        :return:
        """
        for attrib_name in dir(self): # TODO use vars() syntax (built-in) instead, make this cleaner
            attrib_orig = getattr(self, attrib_name)
            if isinstance(attrib_orig, bool) or isinstance(attrib_orig, int):
                continue
            
            # Skip attribute if it has no setter (fset == none)
            if (attrib_name in self.__class__.__dict__.keys()
                    and isinstance(getattr(self.__class__, attrib_name), property)
                    and getattr(self.__class__, attrib_name).fset == None):
                continue
                    
            try:
                setattr(self, attrib_name, sol.value(attrib_orig))
            except NotImplementedError:
                pass
            except AttributeError:
                raise AttributeError("can't set attribute " + attrib_name)
                
            if isinstance(attrib_orig, list):
                try:
                    new_attrib_orig = []
                    for item in attrib_orig:
                        new_attrib_orig.append(item.substitute_solution(sol))
                    setattr(self, attrib_name, new_attrib_orig)
                except:
                    pass
            try:
                setattr(self, attrib_name, attrib_orig.substitute_solution(sol))
            except:
                pass
        return self


class ImplicitAnalysis(AeroSandboxObject):
    def __init__(self):
        """
        If an optimiztion environment is provided, use that. If not, create one.
        """
        args = locals()
        try:
            opti_input = args["opti"]
        except KeyError:
            opti_input = None

        self.opti_provided = opti_input is not None

        if self.opti_provided:
            self.opti = opti_input
        else:
            self.opti = Opti()


class ExplicitAnalysis(AeroSandboxObject):
    pass
