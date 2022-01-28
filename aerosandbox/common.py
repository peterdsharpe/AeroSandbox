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

        def convert(item):
            try:
                return sol.value(item)
            except NotImplementedError:
                pass

            try:
                return item.substitute_solution(sol)
            except AttributeError:
                pass

            if isinstance(item, list) or isinstance(item, tuple):
                return [convert(i) for i in item]

            return item

        for attrib_name in self.__dict__.keys():
            attrib_value = getattr(self, attrib_name)

            if isinstance(attrib_value, bool) or isinstance(attrib_value, int) or isinstance(attrib_value, float):
                continue

            try:
                setattr(self, attrib_name, convert(attrib_value))
                continue
            except (TypeError, AttributeError):
                pass

        return self

    # TODO delete
    # @classmethod
    # def parse_analysis_specific_options(self,
    #                                     analysis_specific_options: Dict[type, Dict[Any, Dict[type, Dict[str, Any]]]]
    #                                     ) -> Dict[type, Dict[Any, Dict[type, Dict[str, Any]]]]:
    #     """
    #     Loops through analysis_specific_options (dict of analysis class: options dict pairs) and calls the
    #     validate_analysis_specific_options method of the invoking AeroSandbox object for the analysis class specified by each key of the dict.
    #
    #     Note: the @classmethod decorator is used because this method is called by the __init__ method of each AeroSandbox object
    #     before the object is instantiated.
    #     """
    #     for analysis, analysis_specific_options_user in analysis_specific_options.items():
    #         analysis_specific_options[analysis] = self.validate_options_for_analysis(analysis, analysis_specific_options_user)
    #
    #     return analysis_specific_options
    #
    #
    # @classmethod
    # def validate_options_for_analysis(self,
    #                                   analysis,
    #                                   options_for_analysis_user: Dict[type, Dict[str, Any]]
    #                                   ) -> Dict[type, Dict[str, Any]]:
    #     """
    #     Validates the user-specified options_for_analysis (dict of parameter: value pairs) for the invoking AeroSandbox object and given analysis class
    #     against a list of options defined within the given analysis class. Returns default values for options not specified by the user according to
    #     defaults defined within the given analysis class.
    #
    #     Note: the @classmethod decorator is used because this method is called by parse_analysis_specific_options, which itself is called in the __init__ method
    #     of each AeroSandbox object before the object is instantiated.
    #     """
    #     options_for_analysis = {
    #             key: analysis.options_for_analysis_defaults[key] for key in analysis.options_for_analysis_by_object[self] # initialize to default options for given object and analysis
    #         }
    #
    #     if options_for_analysis_user:
    #         for key, value in options_for_analysis_user.items():
    #             if key in options_for_analysis.keys():
    #                 options_for_analysis[key] = value
    #             else:
    #                 raise ValueError(f"'{key}' is not a valid option for object {self} within analysis {analysis}. Valid options are: {tuple(options_for_analysis.keys())}")
    #
    #     return options_for_analysis
    #
    # def get_options_for_analysis(self,
    #                              analysis
    #                              ) -> Dict[type, Dict[str, Any]]:
    #     """
    #     Gets the analysis_specific_options for the invoking instantiated AeroSandbox object and given analysis class or returns the default options if
    #     no analysis_specific_options are specified
    #     """
    #     analysis_specific_options = self.analysis_specific_options
    #     if analysis not in analysis_specific_options.keys(): # no options for given analysis
    #         analysis_specific_options = self.validate_options_for_analysis(analysis, {}) # passing an empty dict will return the default options
    #     else:
    #         analysis_specific_options = analysis_specific_options[analysis]
    #
    #     return analysis_specific_options

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

            if not self.opti_provided and not self.opti.x.shape == (0, 1):
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
