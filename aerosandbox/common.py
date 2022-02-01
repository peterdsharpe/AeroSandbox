from aerosandbox.optimization.opti import Opti
from abc import abstractmethod
import copy
from typing import Dict, Any


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


class ExplicitAnalysis(AeroSandboxObject):
    default_analysis_specific_options: Dict[type, Dict[str, Any]] = {}
    """This is part of AeroSandbox's "analysis-specific options" feature, which lets you "tag" geometry objects with 
    flags that change how different analyses act on them. 
    
    This variable, `default_analysis_specific_options`, allows you to specify default values for options that can be used for 
    specific problems. 
    
    This should be a dictionary, where: * keys are the geometry-like types that you might be interested in defining 
    parameters for. * values are dictionaries, where: * keys are strings that label a given option * values are 
    anything. These are used as the default values, in the event that the associated geometry doesn't override those. 
    
    An example of what this variable might look like, for a vortex-lattice method aerodynamic analysis:
    
    >>> default_analysis_specific_options = {
    >>>     Airplane: dict(
    >>>         profile_drag_coefficient=0
    >>>     ),
    >>>     Wing    : dict(
    >>>         wing_level_spanwise_spacing=True,
    >>>         spanwise_resolution=12,
    >>>         spanwise_spacing="cosine",
    >>>         chordwise_resolution=12,
    >>>         chordwise_spacing="cosine",
    >>>         component=None,  # type: int
    >>>         no_wake=False,
    >>>         no_alpha_beta=False,
    >>>         no_load=False,
    >>>         drag_polar=dict(
    >>>             CL1=0,
    >>>             CD1=0,
    >>>             CL2=0,
    >>>             CD2=0,
    >>>             CL3=0,
    >>>             CD3=0,
    >>>         ),
    >>>     )
    >>> }
    
    """

    def get_options(self,
                    geometry_object: AeroSandboxObject,
                    ) -> Dict[str, Any]:
        """
        Retrieves the analysis-specific options that correspond to both:

            * An analysis type (which is this object, "self"), and

            * A specific geometry object, such as an Airplane or Wing.

        Args:
            geometry_object: An instance of an AeroSandbox geometry object, such as an Airplane, Wing, etc.

                * In order for this function to do something useful, you probably want this option to have
                `analysis_specific_options` defined. See the asb.Airplane constructor for an example of this.

        Returns: A dictionary that combines:

            * This analysis's default options for this geometry, if any exist.

            * The geometry's declared analysis-specific-options for this analysis, if it exists. These geometry
            options will override the defaults from the analysis.

            This dictionary has the format:

                * keys are strings, listing specific options

                * values can be any type, and simply state the value of the analysis-specific option following the
                logic above.

        Note: if this analysis defines a set of default options for the geometry type in question (by using
        `self.default_analysis_specific_options`), all keys from the geometry object's `analysis_specific_options`
        will be validated against those in the default options set. A warning will be raised if keys do not
        correspond to those in the defaults, as this (potentially) indicates a typo, which would otherwise be
        difficult to debug.

        """

        ### Determine the types of both this analysis and the geometry object.
        analysis_type: type = self.__class__
        geometry_type: type = geometry_object.__class__

        ### Determine whether this analysis and the geometry object have options that specifically reference each other or not.
        try:
            analysis_options_for_this_geometry = self.default_analysis_specific_options[geometry_type]
            assert hasattr(analysis_options_for_this_geometry, "items")
        except (AttributeError, KeyError, AssertionError):
            analysis_options_for_this_geometry = None

        try:
            geometry_options_for_this_analysis = geometry_object.analysis_specific_options[analysis_type]
            assert hasattr(geometry_options_for_this_analysis, "items")
        except (AttributeError, KeyError, AssertionError):
            geometry_options_for_this_analysis = None

        ### Now, merge those options (with logic depending on whether they exist or not)
        if analysis_options_for_this_geometry is not None:
            options = copy.deepcopy(analysis_options_for_this_geometry)
            if geometry_options_for_this_analysis is not None:
                for k, v in geometry_options_for_this_analysis.items():
                    if k in analysis_options_for_this_geometry.keys():
                        options[k] = v
                    else:

                        import warnings
                        allowable_keys = [f'"{k}"' for k in analysis_options_for_this_geometry.keys()]
                        warnings.warn(
                            f"\nAn object of type '{geometry_type.__name__}' declared the analysis-specific option '{k}' for use with analysis '{analysis_type.__name__}'.\n"
                            f"This was unexpected! Allowable analysis-specific options for '{geometry_type.__name__}' with '{analysis_type.__name__}' are:\n"
                            "\t" + "\n\t".join(allowable_keys) + "\n" "Did you make a typo?",
                            stacklevel=2,
                        )

        else:
            if geometry_options_for_this_analysis is not None:
                options = geometry_options_for_this_analysis
            else:
                options = {}

        return options


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
