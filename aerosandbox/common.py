"""Core base classes for AeroSandbox.

This module defines the fundamental abstract base classes that all AeroSandbox
objects inherit from, including `AeroSandboxObject`, `ExplicitAnalysis`, and
`ImplicitAnalysis`. It also provides utility functions for saving and loading
these objects.
"""
import aerosandbox.numpy as np
from aerosandbox.optimization.opti import Opti, OptiSol
from abc import abstractmethod, ABC
import copy
from typing import Any
import dill
from pathlib import Path
import sys
import warnings


class AeroSandboxObject(ABC):
    """Abstract base class for all AeroSandbox objects.

    This class provides common functionality for AeroSandbox objects, including:

    - Value-based equality comparison (via ``__eq__``)
    - Serialization to/from disk (via ``save`` method and ``load`` function)
    - Shallow and deep copying (via ``copy`` and ``deepcopy`` methods)

    All geometry objects (Airplane, Wing, Airfoil, etc.) and analysis objects
    inherit from this class.

    Notes
    -----
    This is an abstract class and cannot be instantiated directly. Subclasses
    must implement the ``__init__`` method.
    """

    _asb_metadata: dict[str, str] = None

    @abstractmethod
    def __init__(self):
        """Initialize the AeroSandboxObject (abstract method)."""
        pass

    def __eq__(self, other):
        """Check if two AeroSandbox objects are value-equivalent.

        This provides a more sensible default for classes that represent
        physical objects than checking for memory equivalence. Comparison is
        done by checking if the two objects are of the same type and have
        identical ``__dict__`` contents.

        Parameters
        ----------
        other : object
            Another object to compare against.

        Returns
        -------
        bool
            True if the objects are equal (same type and identical attribute
            values), False otherwise.
        """
        if self is other:  # If they point to the same object in memory, they're equal
            return True

        if not isinstance(
            other, type(self)
        ):  # If they are of different types, they cannot be equal
            return False

        if set(self.__dict__.keys()) != set(
            other.__dict__.keys()
        ):  # If they have differing dict keys, don't bother checking values
            return False

        for key in self.__dict__.keys():  # Check equality of all values
            if np.all(self.__dict__[key] == other.__dict__[key]):
                continue
            else:
                return False

        return True

    def save(
        self,
        filename: str | Path | None = None,
        verbose: bool = True,
        automatically_add_extension: bool = True,
    ) -> None:
        """Save the object to a binary file using the ``dill`` library.

        Create a ``.asb`` file that can be loaded with ``aerosandbox.load()``.
        The saved object can be loaded into memory in a different Python
        session or on a different computer, preserving its exact state.

        Parameters
        ----------
        filename : str | Path | None, optional
            The filename to save this object to. Should be a ``.asb`` file.
            If None, uses the object's ``name`` attribute if available,
            otherwise defaults to "untitled".
        verbose : bool, optional
            If True, print a message to the console on successful save.
            Default is True.
        automatically_add_extension : bool, optional
            If True, automatically add the ``.asb`` extension to the filename
            if it doesn't already have it. Default is True.

        Returns
        -------
        None
            Writes the object to disk.
        """
        if filename is None:
            try:
                filename = self.name
            except AttributeError:
                filename = "untitled"

        filename = Path(filename)

        if filename.suffix == "" and automatically_add_extension:
            filename = filename.with_suffix(".asb")

        if verbose:
            print(f"Saving {str(self)} to:\n\t{filename}...")

        import aerosandbox as asb

        self._asb_metadata = {
            "python_version": ".".join(
                [
                    str(sys.version_info.major),
                    str(sys.version_info.minor),
                    str(sys.version_info.micro),
                ]
            ),
            "asb_version": asb.__version__,
        }
        with open(filename, "wb") as f:
            dill.dump(
                obj=self,
                file=f,
            )

    def copy(self):
        """Return a shallow copy of the object."""
        return copy.copy(self)

    def deepcopy(self):
        """Return a deep copy of the object."""
        return copy.deepcopy(self)

    def substitute_solution(
        self,
        sol: OptiSol,
        inplace: bool | None = None,
    ):
        """Substitute a solution from CasADi's solver recursively.

        .. deprecated::
            This function is deprecated and will be removed in a future version.
            Use ``sol(x)``, which now works recursively on complex data structures.

        Parameters
        ----------
        sol : OptiSol
            The solution object from a solved optimization problem.
        inplace : bool | None, optional
            If True, modify this object in-place. If False, return a modified
            copy. Default is True.

        Returns
        -------
        None or AeroSandboxObject
            If ``inplace=True``, returns None (modifies self in-place).
            If ``inplace=False``, returns a new object with substituted values.
        """
        import warnings

        warnings.warn(
            "This function is deprecated and will break at some future point.\n"
            "Use `sol(x)`, which now works recursively on complex data structures.",
            DeprecationWarning,
        )

        # Set defaults
        if inplace is None:
            inplace = True

        def convert(item):
            """Convert CasADi symbolic types to numeric values recursively."""
            # If it can be converted, do the conversion.
            if np.is_casadi_type(item, recursive=False):
                return sol(item)

            t = type(item)

            # If it's a Python iterable, recursively convert it, and preserve the type as best as possible.
            if issubclass(t, list):
                return [convert(i) for i in item]
            if issubclass(t, tuple):
                return tuple([convert(i) for i in item])
            if issubclass(t, set) or issubclass(t, frozenset):
                return {convert(i) for i in item}
            if issubclass(t, dict):
                return {convert(k): convert(v) for k, v in item.items()}

            # Skip certain Python types
            for type_to_skip in (
                bool,
                str,
                int,
                float,
                complex,
                range,
                type(None),
                bytes,
                bytearray,
                memoryview,
            ):
                if issubclass(t, type_to_skip):
                    return item

            # Skip Opti and OptiSol types
            for type_to_skip in (Opti, OptiSol):
                if issubclass(t, type_to_skip):
                    return item

            # If it's any other type, try converting its attribute dictionary:
            try:
                newdict = {k: convert(v) for k, v in item.__dict__.items()}

                if inplace:
                    for k, v in newdict.items():
                        setattr(item, k, v)

                    return item

                else:
                    newitem = copy.copy(item)
                    for k, v in newdict.items():
                        setattr(newitem, k, v)

                    return newitem

            except AttributeError:
                pass

            # Try converting it blindly. This will catch most NumPy-array-like types.
            try:
                return sol(item)
            except (NotImplementedError, TypeError, ValueError):
                pass

            # At this point, we're not really sure what type the object is. Raise a warning and return the item, then hope for the best.
            import warnings

            warnings.warn(
                f"In solution substitution, could not convert an object of type {t}.\n"
                f"Returning it and hoping for the best.",
                UserWarning,
            )

            return item

        if inplace:
            convert(self)

        else:
            return convert(self)


def load(
    filename: str | Path,
    verbose: bool = True,
) -> AeroSandboxObject:
    """Load an AeroSandboxObject from a file.

    Upon loading, compare metadata from the file to the current Python version
    and AeroSandbox version. If there are any discrepancies, raise a warning.

    Parameters
    ----------
    filename : str | Path
        The filename to load from. Should be a ``.asb`` file.
    verbose : bool, optional
        If True, print a message to the console on successful load.
        Default is True.

    Returns
    -------
    AeroSandboxObject
        The deserialized object loaded from the file.

    Warnings
    --------
    UserWarning
        If the Python version or AeroSandbox version used to save the file
        differs from the current versions, a warning is raised about potential
        compatibility issues.
    """
    filename = Path(filename)

    # Load the object from file
    with open(filename, "rb") as f:
        obj = dill.load(f)

    # At this point, the object is loaded
    try:
        metadata = obj._asb_metadata
    except AttributeError:
        warnings.warn(
            "This object was saved without metadata. This may cause compatibility issues.",
            stacklevel=2,
        )
        return obj

    # Check if the Python version is different
    try:
        saved_python_version = metadata["python_version"]
        current_python_version = ".".join(
            [
                str(sys.version_info.major),
                str(sys.version_info.minor),
                str(sys.version_info.micro),
            ]
        )

        saved_python_version_split = saved_python_version.split(".")
        current_python_version_split = current_python_version.split(".")

        if any(
            [
                saved_python_version_split[0] != current_python_version_split[0],
                saved_python_version_split[1] != current_python_version_split[1],
            ]
        ):
            warnings.warn(
                f"This object was saved with Python {saved_python_version}, but you are currently using Python {current_python_version}.\n"
                f"This may cause compatibility issues.",
                stacklevel=2,
            )

    except KeyError:
        warnings.warn(
            "This object was saved without Python version info metadata. This may cause compatibility issues.",
            stacklevel=2,
        )

    # Check if the AeroSandbox version is different
    import aerosandbox as asb

    try:
        saved_asb_version = metadata["asb_version"]

        if saved_asb_version != asb.__version__:
            warnings.warn(
                f"This object was saved with AeroSandbox {saved_asb_version}, but you are currently using AeroSandbox {asb.__version__}.\n"
                f"This may cause compatibility issues.",
                stacklevel=2,
            )

    except KeyError:
        warnings.warn(
            "This object was saved without AeroSandbox version info metadata. This may cause compatibility issues.",
            stacklevel=2,
        )

    if verbose:
        print(f"Loaded {str(obj)} from:\n\t{filename}")

    return obj


class ExplicitAnalysis(AeroSandboxObject):
    """Base class for explicitly-solved analyses.

    An explicit analysis is one that can be computed directly from its inputs
    without requiring iteration or solving a system of equations. Examples
    include simple aerodynamic panel methods or lookup-table-based models.

    This class provides the "analysis-specific options" feature, which allows
    geometry objects to be tagged with flags that change how different analyses
    act on them.

    Attributes
    ----------
    default_analysis_specific_options : dict[type, dict[str, Any]]
        Default values for analysis-specific options, keyed by geometry type.
        Subclasses should override this to specify their default options.

    Examples
    --------
    An example of what ``default_analysis_specific_options`` might look like
    for a vortex-lattice method aerodynamic analysis:

    >>> default_analysis_specific_options = {
    ...     Airplane: dict(
    ...         profile_drag_coefficient=0
    ...     ),
    ...     Wing: dict(
    ...         spanwise_resolution=12,
    ...         spanwise_spacing="cosine",
    ...         chordwise_resolution=12,
    ...         chordwise_spacing="cosine",
    ...     )
    ... }
    """

    default_analysis_specific_options: dict[type, dict[str, Any]] = {}

    def get_options(
        self,
        geometry_object: AeroSandboxObject,
    ) -> dict[str, Any]:
        """Retrieve analysis-specific options for a geometry object.

        Combine this analysis's default options for the given geometry type
        with any options declared on the geometry object itself. Geometry
        options override the analysis defaults.

        Parameters
        ----------
        geometry_object : AeroSandboxObject
            An instance of an AeroSandbox geometry object, such as an Airplane,
            Wing, etc. For this function to be useful, the geometry object
            should have ``analysis_specific_options`` defined. See the
            ``asb.Airplane`` constructor for an example.

        Returns
        -------
        dict[str, Any]
            A dictionary combining this analysis's default options with the
            geometry's declared options. Keys are option names (strings),
            and values are the option values.

        Warnings
        --------
        UserWarning
            If the geometry object declares an analysis-specific option that
            is not in this analysis's defaults, a warning is raised (this
            potentially indicates a typo).
        """
        ### Determine the types of both this analysis and the geometry object.
        analysis_type: type = self.__class__
        geometry_type: type = geometry_object.__class__

        ### Determine whether this analysis and the geometry object have options that specifically reference each other or not.
        try:
            analysis_options_for_this_geometry = self.default_analysis_specific_options[
                geometry_type
            ]
            assert hasattr(analysis_options_for_this_geometry, "items")
        except (AttributeError, KeyError, AssertionError):
            analysis_options_for_this_geometry = None

        try:
            geometry_options_for_this_analysis = (
                geometry_object.analysis_specific_options[analysis_type]
            )
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

                        allowable_keys = [
                            f'"{k}"' for k in analysis_options_for_this_geometry.keys()
                        ]
                        warnings.warn(
                            f"\nAn object of type '{geometry_type.__name__}' declared the analysis-specific option '{k}' for use with analysis '{analysis_type.__name__}'.\n"
                            f"This was unexpected! Allowable analysis-specific options for '{geometry_type.__name__}' with '{analysis_type.__name__}' are:\n"
                            "\t" + "\n\t".join(allowable_keys) + "\n"
                            "Did you make a typo?",
                            stacklevel=2,
                        )

        else:
            if geometry_options_for_this_analysis is not None:
                options = geometry_options_for_this_analysis
            else:
                options = {}

        return options


class ImplicitAnalysis(AeroSandboxObject):
    """Base class for implicitly-solved analyses.

    An implicit analysis requires solving a system of equations (e.g., via
    iteration or numerical optimization) to determine the solution. Examples
    include nonlinear aerodynamic solvers or coupled aerostructural analyses.

    This class provides infrastructure for working with AeroSandbox's ``Opti``
    optimization environment. Subclasses should use the ``@ImplicitAnalysis.initialize``
    decorator on their ``__init__`` methods.

    Notes
    -----
    The key difference from ``ExplicitAnalysis`` is that implicit analyses
    require an optimization environment (``asb.Opti``) to solve. If the user
    doesn't provide one, a new one is created and the problem is solved
    automatically. If the user provides an existing ``Opti`` environment, the
    analysis is added to it but not solved (allowing multi-disciplinary coupling).
    """

    @staticmethod
    def initialize(init_method):
        """Decorate the ``__init__`` method of ImplicitAnalysis subclasses.

        This decorator ensures that every ImplicitAnalysis has an ``opti``
        property pointing to an optimization environment (``asb.Opti``).

        Parameters
        ----------
        init_method : callable
            The ``__init__`` method to be decorated.

        Returns
        -------
        callable
            The wrapped ``__init__`` method with ``opti`` parameter handling.

        Examples
        --------
        >>> class MyAnalysis(ImplicitAnalysis):
        ...
        ...     @ImplicitAnalysis.initialize
        ...     def __init__(self):
        ...         self.a = self.opti.variable(init_guess=1)
        ...         self.b = self.opti.variable(init_guess=2)
        ...         self.opti.subject_to(self.a == self.b ** 2)

        Notes
        -----
        The decorator adds an optional ``opti`` parameter to the ``__init__``
        method:

        - If ``opti`` is not provided, a new ``asb.Opti`` environment is
          created, the analysis is solved, and results are substituted in-place.
        - If ``opti`` is provided, the analysis is added to that environment
          but not solved (user must call ``opti.solve()`` later).

        A property ``opti_provided`` is also set to indicate which case applies.
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
                self.__dict__ = sol(self.__dict__)

        return init_wrapped

    class ImplicitAnalysisInitError(Exception):
        """Exception raised when an ImplicitAnalysis is not properly initialized.

        This error occurs when an ImplicitAnalysis subclass's ``__init__`` method
        was not decorated with ``@ImplicitAnalysis.initialize``.
        """

        def __init__(
            self,
            message="""
    Your ImplicitAnalysis object doesn't have an `opti` property!
    This is almost certainly because you didn't decorate your object's __init__ method with 
    `@ImplicitAnalysis.initialize`, which you should go do.
                     """,
        ):
            self.message = message
            super().__init__(self.message)

    @property
    def opti(self):
        """Return the optimization environment for this analysis.

        Returns
        -------
        Opti
            The ``asb.Opti`` optimization environment.

        Raises
        ------
        ImplicitAnalysisInitError
            If the ``__init__`` method was not decorated with
            ``@ImplicitAnalysis.initialize``.
        """
        try:
            return self._opti
        except AttributeError:
            raise self.ImplicitAnalysisInitError()

    @opti.setter
    def opti(self, value: Opti):
        self._opti = value

    @property
    def opti_provided(self):
        """Return whether an ``Opti`` environment was provided by the user.

        Returns
        -------
        bool
            True if the user provided an ``Opti`` environment, False if one
            was created automatically.

        Raises
        ------
        ImplicitAnalysisInitError
            If the ``__init__`` method was not decorated with
            ``@ImplicitAnalysis.initialize``.
        """
        try:
            return self._opti_provided
        except AttributeError:
            raise self.ImplicitAnalysisInitError()

    @opti_provided.setter
    def opti_provided(self, value: bool):
        self._opti_provided = value
