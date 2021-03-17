from aerosandbox.optimization.opti import Opti
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
        Substitutes a solution from CasADi's solver recursively as an in-place operation.

        In-place operation. To make it not in-place, do `y = copy.deepcopy(x)` or similar first.
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

    @staticmethod
    def initialize(init_method):
        """
        A decorator that should be applied to the __init__ method of ImplicitAnalysis or any subclass of it.

        Usage example:

        >>> class MyAnalysis(ImplicitAnalysis):
        >>>
        >>>     @ImplicitAnalysis.initialize
        >>>     def __init__(self):
        >>>         self.a = self.opti.variable(init_guess = 1)
        >>>         self.b = self.opti.variable(init_guess = 2)
        >>>
        >>>         self.opti.subject_to(
        >>>             self.a == self.b ** 2
        >>>         ) # Add a nonlinear governing equation

        Functionality:

        The basic purpose of this wrapper is to ensure that every ImplicitAnalysis has an `opti` property that points to
        an optimization environment (asb.Opti type) that it can work in.

        How do we obtain an asb.Opti environment to work in? Well, this decorator adds an optional `opti` parameter to
        the __init__ method that it is applied to.

            1. If this `opti` parameter is not provided, then a new empty `asb.Opti` environment is created and stored as
            `ImplicitAnalysis.opti`.

            2. If the `opti` parameter is provided, then we simply assign the given `asb.Opti` environment (which may
            already contain other variables/constraints/objective) to `ImplicitAnalysis.opti`.

        In addition, a property called `ImplicitAnalysis.opti_provided` is stored, which records whether the user
        provided an Opti environment or if one was instead created for them.

        If the user did not provide an Opti environment (Option 1 from our list above), we assume that the user basically
        just wants to perform a normal, single-disciplinary analysis. So, in this case, we proceed to solve the analysis as-is
        and do an in-place substitution of the solution.

        If the user did provide an Opti environment (Option 2 from our list above), we assume that the user might potentially want
        to add other implicit analyses to the problem. So, in this case, we don't solve the analysis, and the user must later
        solve the analysis by calling `sol = opti.solve()` or similar.

        """

        def init_wrapped(self, *args, opti=None, **kwargs):
            if opti is None:
                self.opti = Opti()
                self.opti_provided = False
            else:
                self.opti = opti
                self.opti_provided = True

            init_method(self, *args, **kwargs)

            if not self.opti_provided:
                sol = self.opti.solve()
                self.substitute_solution(sol)

        return init_wrapped

    class ImplicitAnalysisInitError(Exception):
        def __init__(self,
                     message="""
    Your ImplicitAnalysis object doesn't have an `opti` property!
    This is almost certainly because you didn't decorate your object's __init__ method with 
    `@ImplicitAnalysis.initialize`, which you should go do.
                     """
                     ):
            self.message = message
            super().__init__(self.message)

    @property
    @abstractmethod
    def opti(self):
        try:
            return self._opti
        except AttributeError:
            raise self.ImplicitAnalysisInitError()

    @opti.setter
    @abstractmethod
    def opti(self, value: Opti):
        self._opti = value

    @property
    @abstractmethod
    def opti_provided(self):
        try:
            return self._opti_provided
        except AttributeError:
            raise self.ImplicitAnalysisInitError()

    @opti_provided.setter
    @abstractmethod
    def opti_provided(self, value: bool):
        self._opti_provided = value


class ExplicitAnalysis(AeroSandboxObject):
    pass
