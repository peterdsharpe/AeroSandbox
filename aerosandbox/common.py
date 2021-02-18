from aerosandbox.optimization import Opti
from abc import abstractmethod


class AeroSandboxObject:

    @abstractmethod
    def __init__(self):
        """
        Denotes AeroSandboxObject as an abstract class, meaning you can't instantiate it directly - you must subclass
        (extend) it instead.
        """
        pass

    def substitute_solution(self, sol):
        """
        Substitutes a solution from CasADi's solver.

        In-place operation. To make it not in-place, do a copy.deepcopy(x) first.
        :param sol: OptiSol object.
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
            try:
                setattr(self, attrib_name, attrib_orig.substitute_solution(sol))
            except:
                pass
        return self


class ImplicitAnalysis(AeroSandboxObject):
    """
    An implicit analysis. To extend this, inherit from this. Your new class should look something like:

    >>> class MyImplicitAnalysis(ImplicitAnalysis):
    >>>
    >>>     def __init__(self, *args):
    >>>
    >>>         super().__init__()
    >>>
    >>>         do_my_entire_analysis_here(*args) # Outputs should be stored as class properties, e.g. `self.lift_force`
    >>>
    >>>         super()._init_end()

    See the LiftingLine class as an example of this.

    """

    def __init__(self):  # TODO make this an abstractmethod after verifying functionality.
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

    def _init_end(self):
        if not self.opti_provided:
            sol = self.opti.solve()
            self.substitute_solution(sol)


class ExplicitAnalysis(AeroSandboxObject):
    pass
