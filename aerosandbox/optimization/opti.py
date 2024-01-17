from typing import Union, List, Dict, Callable, Any, Tuple, Set, Optional
import json
import casadi as cas
import aerosandbox.numpy as np
from aerosandbox.tools import inspect_tools
from sortedcontainers import SortedDict
import copy


class Opti(cas.Opti):
    """
    The base class for mathematical optimization. For detailed usage, see the docstrings in its key methods:
        * Opti.variable()
        * Opti.subject_to()
        * Opti.parameter()
        * Opti.solve()

    Example usage is as follows:

    >>> opti = asb.Opti() # Initializes an optimization environment
    >>> x = opti.variable(init_guess=5) # Initializes a new variable in that environment
    >>> f = x ** 2 # Evaluates a (in this case, nonlinear) function based on a variable
    >>> opti.subject_to(x > 3) # Adds a constraint to be enforced
    >>> opti.minimize(f) # Sets the objective function as f
    >>> sol = opti.solve() # Solves the problem using CasADi and IPOPT backend
    >>> print(sol.value(x)) # Prints the value of x at the optimum.
    """

    def __init__(self,
                 variable_categories_to_freeze: Union[List[str], str] = None,
                 cache_filename: str = None,
                 load_frozen_variables_from_cache: bool = False,
                 save_to_cache_on_solve: bool = False,
                 ignore_violated_parametric_constraints: bool = False,
                 freeze_style: str = "parameter",
                 ):  # TODO document

        # Default arguments
        if variable_categories_to_freeze is None:
            variable_categories_to_freeze = []

        # Parent class initialization
        super().__init__()

        # Initialize class variables
        self.variable_categories_to_freeze = variable_categories_to_freeze
        self.cache_filename = cache_filename
        self.load_frozen_variables_from_cache = load_frozen_variables_from_cache  # TODO load and start tracking
        self.save_to_cache_on_solve = save_to_cache_on_solve
        self.ignore_violated_parametric_constraints = ignore_violated_parametric_constraints
        self.freeze_style = freeze_style

        # Start tracking variables and categorize them.
        self.variables_categorized = {}  # category name [str] : list of variables [list]

        # Track variable declaration locations, useful for debugging
        self._variable_declarations = SortedDict()  # first index in super().x : (filename, lineno, code_context, n_vars)
        self._constraint_declarations = SortedDict()  # first index in super().g : (filename, lineno, code_context, n_cons)
        self._variable_index_counter = 0
        self._constraint_index_counter = 0

    ### Primary Methods

    def variable(self,
                 init_guess: Union[float, np.ndarray] = None,
                 n_vars: int = None,
                 scale: float = None,
                 freeze: bool = False,
                 log_transform: bool = False,
                 category: str = "Uncategorized",
                 lower_bound: float = None,
                 upper_bound: float = None,
                 _stacklevel: int = 1,
                 ) -> cas.MX:
        """
        Initializes a new decision variable (or vector of decision variables). You should pass an initial guess (
        `init_guess`) upon defining a new variable. Dimensionality is inferred from this initial guess, but it can be
        overridden; see below for syntax.

        It is highly, highly recommended that you provide a scale (`scale`) for each variable, especially for
        nonconvex problems, although this is not strictly required.

        Usage notes:

            When using vector variables, individual components of this vector of variables can be accessed via normal
            indexing. Example:
                >>> opti = asb.Opti()
                >>> my_var = opti.variable(n_vars = 5)
                >>> opti.subject_to(my_var[3] >= my_var[2])  # This is a valid way of indexing
                >>> my_sum = asb.sum(my_var)  # This will sum up all elements of `my_var`

        Args:

            init_guess: Initial guess for the optimal value of the variable being initialized. This is where in the
                design space the optimizer will start looking.

                This can be either a float or a NumPy ndarray; the dimension of the variable (i.e. scalar,
                vector) that is created will be automatically inferred from the shape of the initial guess you
                provide here. (Although it can also be overridden using the `n_vars` parameter; see below.)

                For scalar variables, your initial guess should be a float:

                >>> opti = asb.Opti()
                >>> scalar_var = opti.variable(init_guess=5) # Initializes a scalar variable at a value of 5

                For vector variables, your initial guess should be either:

                    * a float, in which case you must pass the length of the vector as `n_vars`, otherwise a scalar
                    variable will be created:

                    >>> opti = asb.Opti()
                    >>> vector_var = opti.variable(init_guess=5, n_vars=10) # Initializes a vector variable of length
                    >>> # 10, with all 10 elements set to an initial guess of 5.

                    * a NumPy ndarray, in which case each element will be initialized to the corresponding value in
                    the given array:

                    >>> opti = asb.Opti()
                    >>> vector_var = opti.variable(init_guess=np.linspace(0, 5, 10)) # Initializes a vector variable of
                    >>> # length 10, with all 10 elements initialized to linearly vary between 0 and 5.

                In the case where the variable is to be log-transformed (see `log_transform`), the initial guess
                should not be log-transformed as well - just supply the initial guess as usual. (Log-transform of the
                initial guess happens under the hood.) The initial guess must, of course, be a positive number in
                this case.

            n_vars: [Optional] Used to manually override the dimensionality of the variable to create; if not
                provided, the dimensionality of the variable is inferred from the initial guess `init_guess`.

                The only real case where you need to use this argument would be if you are initializing a vector
                variable to a scalar value, but you don't feel like using `init_guess=value * np.ones(n_vars)`.
                For example:

                    >>> opti = asb.Opti()
                    >>> vector_var = opti.variable(init_guess=5, n_vars=10) # Initializes a vector variable of length
                    >>> # 10, with all 10 elements set to an initial guess of 5.

            scale: [Optional] Approximate scale of the variable.

                For example, if you're optimizing the design of a automobile and setting the tire diameter as an
                optimization variable, you might choose `scale=0.5`, corresponding to 0.5 meters.

                Properly scaling your variables can have a huge impact on solution speed (or even if the optimizer
                converges at all). Although most modern second-order optimizers (such as IPOPT, used here) are
                theoretically scale-invariant, numerical precision issues due to floating-point arithmetic can make
                solving poorly-scaled problems really difficult or impossible. See here for more info:
                https://web.casadi.org/blog/nlp-scaling/

                If not specified, the code will try to pick a sensible value by defaulting to the `init_guess`.

            freeze: [Optional] This boolean tells the optimizer to "freeze" the variable at a specific value. In
                order to select the determine to freeze the variable at, the optimizer will use the following logic:

                    * If you initialize a new variable with the parameter `freeze=True`: the optimizer will freeze
                    the variable at the value of initial guess.

                        >>> opti = Opti()
                        >>> my_var = opti.variable(init_guess=5, freeze=True) # This will freeze my_var at a value of 5.

                    * If the Opti instance is associated with a cache file, and you told it to freeze a specific
                    category(s) of variables that your variable is a member of, and you didn't manually specify to
                    freeze the variable: the variable will be frozen based on the value in the cache file (and ignore
                    the `init_guess`). Example:

                        >>> opti = Opti(cache_filename="my_file.json", variable_categories_to_freeze=["Wheel Sizing"])
                        >>> # Assume, for example, that `my_file.json` was from a previous run where my_var=10.
                        >>> my_var = opti.variable(init_guess=5, category="Wheel Sizing")
                        >>> # This will freeze my_var at a value of 10 (from the cache file, not the init_guess)

                    * If the Opti instance is associated with a cache file, and you told it to freeze a specific
                    category(s) of variables that your variable is a member of, but you then manually specified that
                    the variable should be frozen: the variable will once again be frozen at the value of `init_guess`:

                        >>> opti = Opti(cache_filename="my_file.json", variable_categories_to_freeze=["Wheel Sizing"])
                        >>> # Assume, for example, that `my_file.json` was from a previous run where my_var=10.
                        >>> my_var = opti.variable(init_guess=5, category="Wheel Sizing", freeze=True)
                        >>> # This will freeze my_var at a value of 5 (`freeze` overrides category loading.)

                Motivation for freezing variables:

                    The ability to freeze variables is exceptionally useful when designing engineering systems. Let's say
                    we're designing an airplane. In the beginning of the design process, we're doing "clean-sheet" design
                    - any variable is up for grabs for us to optimize on, because the airplane doesn't exist yet!
                    However, the farther we get into the design process, the more things get "locked in" - we may have
                    ordered jigs, settled on a wingspan, chosen an engine, et cetera. So, if something changes later (
                    let's say that we discover that one of our assumptions was too optimistic halfway through the design
                    process), we have to make up for that lost margin using only the variables that are still free. To do
                    this, we would freeze the variables that are already decided on.

                    By categorizing variables, you can also freeze entire categories of variables. For example,
                    you can freeze all of the wing design variables for an airplane but leave all of the fuselage
                    variables free.

                    This idea of freezing variables can also be used to look at off-design performance - freeze a
                    design, but change the operating conditions.

            log_transform: [Optional] Advanced use only. A flag of whether to internally-log-transform this variable
                before passing it to the optimizer. Good for known positive engineering quantities that become nonsensical
                if negative (e.g. mass). Log-transforming these variables can also help maintain convexity.

            category: [Optional] What category of variables does this belong to? # TODO expand docs

            lower_bound: [Optional] If provided, defines a bounds constraint on the new variable that keeps the
                variable above a given value.

            upper_bound: [Optional] If provided, defines a bounds constraint on the new variable that keeps the
                variable below a given value.

            _stacklevel: Optional and advanced, purely used for debugging. Allows users to correctly track where
                variables are declared in the event that they are subclassing `aerosandbox.Opti`. Modifies the
                stacklevel of the declaration tracked, which is then presented using
                `aerosandbox.Opti.variable_declaration()`.

        Returns:
            The variable itself as a symbolic CasADi variable (MX type).

        """
        ### Set defaults
        if init_guess is None:
            import warnings
            if log_transform:
                init_guess = 1
                warnings.warn("No initial guess set for Opti.variable(). Defaulting to 1 (log-transformed variable).",
                              stacklevel=2)
            else:
                init_guess = 0
                warnings.warn("No initial guess set for Opti.variable(). Defaulting to 0.", stacklevel=2)
        if n_vars is None:  # Infer dimensionality from init_guess if it is not provided
            n_vars = np.length(init_guess)
        if scale is None:  # Infer a scale from init_guess if it is not provided
            if log_transform:
                scale = 1
            else:
                scale = np.mean(np.fabs(init_guess))  # Initialize the scale to a heuristic based on the init_guess
                if isinstance(scale,
                              cas.MX) or scale == 0:  # If that heuristic leads to a scale of 0, use a scale of 1 instead.
                    scale = 1

                # scale = np.fabs(
                #     np.where(
                #         init_guess != 0,
                #         init_guess,
                #         1
                #     ))

        length_init_guess = np.length(init_guess)

        if length_init_guess != 1 and length_init_guess != n_vars:
            raise ValueError(f"`init_guess` has length {length_init_guess}, but `n_vars` is {n_vars}!")

        # Try to convert init_guess to a float or np.ndarray if it is an Opti parameter.
        try:
            init_guess = self.value(init_guess)
        except RuntimeError as e:
            if not (
                    freeze and self.freeze_style == "float"
            ):
                raise TypeError(
                    "The `init_guess` for a new Opti variable must not be a function of an existing Opti variable."
                )

        # Validate the inputs
        if log_transform:
            if np.any(init_guess <= 0):
                raise ValueError(
                    "If you are initializing a log-transformed variable, the initial guess(es) must all be positive.")
        if np.any(scale <= 0):
            raise ValueError("The 'scale' argument must be a positive number.")

        # If the variable is in a category to be frozen, fix the variable at the initial guess.
        is_manually_frozen = freeze
        if (
                category in self.variable_categories_to_freeze or
                category == self.variable_categories_to_freeze or
                self.variable_categories_to_freeze == "all"
        ):
            freeze = True

        # If the variable is to be frozen, return the initial guess. Otherwise, define the variable using CasADi symbolics.
        if freeze:
            if self.freeze_style == "parameter":
                var = self.parameter(n_params=n_vars, value=init_guess)
            elif self.freeze_style == "float":
                if n_vars == 1:
                    var = init_guess
                else:
                    var = init_guess * np.ones(n_vars)
            else:
                raise ValueError("Bad value of `Opti.freeze_style`!")
        else:
            if not log_transform:
                var = scale * super().variable(n_vars)
                self.set_initial(var, init_guess)
            else:
                log_scale = scale / init_guess
                log_var = log_scale * super().variable(n_vars)
                var = np.exp(log_var)
                self.set_initial(log_var, np.log(init_guess))

            # Track where this variable was declared in code.
            filename, lineno, code_context = inspect_tools.get_caller_source_location(stacklevel=_stacklevel + 1)
            self._variable_declarations[self._variable_index_counter] = (
                filename,
                lineno,
                code_context,
                n_vars
            )
            self._variable_index_counter += n_vars

        # Track the category of the variable
        if category not in self.variables_categorized:  # Add a category if it does not exist
            self.variables_categorized[category] = []
        self.variables_categorized[category].append(var)
        try:
            var.is_manually_frozen = is_manually_frozen
        except AttributeError:
            pass

        # Apply bounds
        if not (freeze and self.ignore_violated_parametric_constraints):
            if (not log_transform) or (freeze):
                if lower_bound is not None:
                    self.subject_to(
                        var / scale >= lower_bound / scale,
                        _stacklevel=_stacklevel + 1
                    )
                if upper_bound is not None:
                    self.subject_to(
                        var / scale <= upper_bound / scale,
                        _stacklevel=_stacklevel + 1
                    )
            else:
                if lower_bound is not None:
                    self.subject_to(
                        log_var / log_scale >= np.log(lower_bound) / log_scale,
                        _stacklevel=_stacklevel + 1
                    )
                if upper_bound is not None:
                    self.subject_to(
                        log_var / log_scale <= np.log(upper_bound) / log_scale,
                        _stacklevel=_stacklevel + 1
                    )

        return var

    def subject_to(self,
                   constraint: Union[cas.MX, bool, List],  # TODO add scale
                   _stacklevel: int = 1,
                   ) -> Union[cas.MX, None, List[cas.MX]]:
        """
        Initialize a new equality or inequality constraint(s).

        Args:
            constraint: A constraint that you want to hold true at the optimum.

                Inequality example:

                >>> x = opti.variable()
                >>> opti.subject_to(x >= 5)

                Equality example; also showing that you can directly constrain functions of variables:

                >>> x = opti.variable()
                >>> f = np.sin(x)
                >>> opti.subject_to(f == 0.5)

                You can also pass in a list of multiple constraints using list syntax. For example:

                >>> x = opti.variable()
                >>> opti.subject_to([
                >>>     x >= 5,
                >>>     x <= 10
                >>> ])

            _stacklevel: Optional and advanced, purely used for debugging. Allows users to correctly track where
            constraints are declared in the event that they are subclassing `aerosandbox.Opti`. Modifies the
            stacklevel of the declaration tracked, which is then presented using
            `aerosandbox.Opti.constraint_declaration()`.

        Returns:
            The dual variable associated with the new constraint. If the `constraint` input is a list, returns
            a list of dual variables.

        """
        # Determine whether you're dealing with a single (possibly vectorized) constraint or a list of constraints.
        # If the latter, recursively apply them.
        if type(constraint) in (list, tuple):
            return [
                self.subject_to(each_constraint, _stacklevel=_stacklevel + 2)  # return the dual of each constraint
                for each_constraint in constraint
            ]

        # If it's a proper constraint (MX-type and non-parametric),
        # pass it into the parent class Opti formulation and be done with it.
        if isinstance(constraint, cas.MX) and not self.advanced.is_parametric(constraint):
            # constraint = cas.cse(constraint)
            super().subject_to(constraint)
            dual = self.dual(constraint)

            # Track where this constraint was declared in code.
            n_cons = np.length(constraint)
            filename, lineno, code_context = inspect_tools.get_caller_source_location(stacklevel=_stacklevel + 1)
            self._constraint_declarations[self._constraint_index_counter] = (
                filename,
                lineno,
                code_context,
                n_cons
            )
            self._constraint_index_counter += np.length(constraint)

            return dual
        else:  # Constraint is not valid because it is not MX type or is parametric.
            try:
                constraint_satisfied = np.all(self.value(constraint))  # Determine if the constraint is true
            except Exception:
                raise TypeError(f"""Opti.subject_to could not determine the truthiness of your constraint, and it
                    doesn't appear to be a symbolic type or a boolean type. You supplied the following constraint:
                    {constraint}""")

            if isinstance(constraint,
                          cas.MX) and not constraint_satisfied:  # Determine if the constraint is *almost* true
                try:
                    LHS = constraint.dep(0)
                    RHS = constraint.dep(1)
                    LHS_value = self.value(LHS)
                    RHS_value = self.value(RHS)
                except Exception:
                    raise ValueError(
                        """Could not evaluate the LHS and RHS of the constraint - are you sure you passed in a comparative expression?""")

                constraint_satisfied = np.allclose(LHS_value,
                                                   RHS_value)  # Call the constraint satisfied if it is *almost* true.

            if constraint_satisfied or self.ignore_violated_parametric_constraints:
                # If the constraint(s) always evaluates True (e.g. if you enter "5 > 3"), skip it.
                # This allows you to toggle frozen variables without causing problems with setting up constraints.
                return None  # dual of an always-true constraint doesn't make sense to evaluate.
            else:
                # If any of the constraint(s) are always False (e.g. if you enter "5 < 3"), raise an error.
                # This indicates that the problem is infeasible as-written, likely because the user has frozen too
                # many decision variables using the Opti.variable(freeze=True) syntax.
                raise RuntimeError(f"""The problem is infeasible due to a constraint that always evaluates False. 
                This can happen if you've frozen too many decision variables, leading to an overconstrained problem.""")

    def minimize(self,
                 f: cas.MX,
                 ) -> None:
        # f = cas.cse(f)
        super().minimize(f)

    def maximize(self,
                 f: cas.MX,
                 ) -> None:
        # f = cas.cse(f)
        super().minimize(-1 * f)

    def parameter(self,
                  value: Union[float, np.ndarray] = 0.,
                  n_params: int = None,
                  ) -> cas.MX:
        """
        Initializes a new parameter (or vector of parameters). You must pass a value (`value`) upon defining a new
        parameter. Dimensionality is inferred from this value, but it can be overridden; see below for syntax.

        Args:

            value: Value to set the new parameter to.

                This can either be a float or a NumPy ndarray; the dimension of the parameter (i.e. scalar,
                vector) that is created will be automatically inferred from the shape of the value you provide here.
                (Although it can be overridden using the `n_params` parameter; see below.)

                For scalar parameters, your value should be a float:
                >>> opti = asb.Opti()
                >>> scalar_param = opti.parameter(value=5) # Initializes a scalar parameter and sets its value to 5.

                For vector variables, your value should be either:

                    * a float, in which case you must pass the length of the vector as `n_params`, otherwise a scalar
                    parameter will be created:

                    >>> opti = asb.Opti()
                    >>> vector_param = opti.parameter(value=5, n_params=10) # Initializes a vector parameter of length
                    >>> # 10, with all 10 elements set to value of 5.

                    * a NumPy ndarray, in which case each element will be set to the corresponding value in the given
                    array:

                    >>> opti = asb.Opti()
                    >>> vector_param = opti.parameter(value=np.linspace(0, 5, 10)) # Initializes a vector parameter of
                    >>> # length 10, with all 10 elements set to a value varying from 0 to 5.

            n_params: [Optional] Used to manually override the dimensionality of the parameter to create; if not
                provided, the dimensionality of the parameter is inferred from `value`.

                The only real case where you need to use this argument would be if you are initializing a vector
                parameter to a scalar value, but you don't feel like using `value=my_value * np.ones(n_vars)`.
                For example:

                    >>> opti = asb.Opti()
                    >>> vector_param = opti.parameter(value=5, n_params=10) # Initializes a vector parameter of length
                    >>> # 10, with all 10 elements set to a value of 5.

        Returns:
            The parameter itself as a symbolic CasADi variable (MX type).

        """
        # Infer dimensionality from value if it is not provided
        if n_params is None:
            n_params = np.length(value)

        # Create the parameter
        param = super().parameter(n_params)

        # Set the value of the parameter
        self.set_value(param, value)

        return param

    def solve(self,
              parameter_mapping: Dict[cas.MX, float] = None,
              max_iter: int = 1000,
              max_runtime: float = 1e20,
              callback: Callable[[int], Any] = None,
              verbose: bool = True,
              jit: bool = False,  # TODO document, add unit tests for jit
              detect_simple_bounds: bool = False,  # TODO document
              options: Dict = None,  # TODO document
              behavior_on_failure: str = "raise",
              ) -> "OptiSol":
        """
        Solve the optimization problem using CasADi with IPOPT backend.

        Args:
            parameter_mapping: [Optional] Allows you to specify values for parameters.
                Dictionary where the key is the parameter and the value is the value to be set to.

                Example: # TODO update syntax for required init_guess
                    >>> opti = asb.Opti()
                    >>> x = opti.variable()
                    >>> p = opti.parameter()
                    >>> opti.minimize(x ** 2)
                    >>> opti.subject_to(x >= p)
                    >>> sol = opti.solve(
                    >>>     {
                    >>>         p: 5 # Sets the value of parameter p to 5, then solves.
                    >>>     }
                    >>> )

            max_iter: [Optional] The maximum number of iterations allowed before giving up.

            max_runtime: [Optional] Gives the maximum allowable runtime before giving up.

            callback: [Optional] A function to be called at each iteration of the optimization algorithm.
                Useful for printing progress or displaying intermediate results.

                The callback function `func` should have the syntax `func(iteration_number)`, where iteration_number
                is an integer corresponding to the current iteration number. In order to access intermediate
                quantities of optimization variables (e.g. for plotting), use the `Opti.debug.value(x)` syntax for
                each variable `x`.

            verbose: Controls the verbosity of the solver. If True, IPOPT will print its progress to the console.

            jit: Experimental. If True, the optimization problem will be compiled to C++ and then JIT-compiled
                using the CasADi JIT compiler. This can lead to significant speedups, but may also lead to
                unexpected behavior, and may not work on all platforms.

            options: [Optional] A dictionary of options to pass to IPOPT. See the IPOPT documentation for a list of
                available options.

            behavior_on_failure: [Optional] What should we do if the optimization fails? Options are:

                * "raise": Raise an exception. This is the default behavior.

                * "return_last": Returns the solution from the last iteration, and raise a warning.

                    NOTE: The returned solution may not be feasible! (It also may not be optimal.)

        Returns: An OptiSol object that contains the solved optimization problem. To extract values, use
            my_optisol(variable).

            Example:
                >>> sol = opti.solve()
                >>> x_opt = sol(x) # Get the value of variable x at the optimum.

        """
        if parameter_mapping is None:
            parameter_mapping = {}

        ### If you're loading frozen variables from cache, do it here:
        if self.load_frozen_variables_from_cache:
            solution_dict = self.get_solution_dict_from_cache()
            for category in self.variable_categories_to_freeze:
                category_variables = self.variables_categorized[category]
                category_values = solution_dict[category]

                if len(category_variables) != len(category_values):
                    raise RuntimeError("""Problem with loading cached solution: it looks like new variables have been
                    defined since the cached solution was saved (or variables were defined in a different order). 
                    Because of this, the cache cannot be loaded. 
                    Re-run the original optimization study to regenerate the cached solution.""")

                for var, val in zip(category_variables, category_values):
                    if not var.is_manually_frozen:
                        parameter_mapping = {
                            **parameter_mapping,
                            var: val
                        }

        ### Map any parameters to needed values
        for k, v in parameter_mapping.items():
            if not np.is_casadi_type(k, recursive=False):
                raise TypeError(
                    f"All keys in `parameter_mapping` must be CasADi parameters; you gave an object of type \'{type(k).__name__}\'.\n"
                    f"In general, make sure all keys are the result of calling `opti.parameter()`.")

            size_k = np.prod(k.shape)
            try:
                size_v = np.prod(v.shape)
            except AttributeError:
                size_v = 1
            if size_k != size_v:
                raise RuntimeError("""Problem with loading cached solution: it looks like the length of a vectorized 
                variable has changed since the cached solution was saved (or variables were defined in a different order). 
                Because of this, the cache cannot be loaded. 
                Re-run the original optimization study to regenerate the cached solution.""")

            self.set_value(k, v)

        ### Set solver settings.
        if options is None:
            options = {}

        default_options = {
            "ipopt.sb"                   : 'yes',  # Hide the IPOPT banner.
            "ipopt.max_iter"             : max_iter,
            "ipopt.max_cpu_time"         : max_runtime,
            "ipopt.mu_strategy"          : "adaptive",
            "ipopt.fast_step_computation": "yes",
            "detect_simple_bounds"       : detect_simple_bounds,
        }

        if jit:
            default_options["jit"] = True
            # options["compiler"] = "shell"  # Recommended by CasADi devs, but doesn't work on my machine
            default_options["jit_options"] = {
                "flags": ["-O3"],
                # "verbose": True
            }

        if verbose:
            default_options["ipopt.print_level"] = 5  # Verbose, per-iteration printing.
        else:
            default_options["print_time"] = False  # No time printing
            default_options["ipopt.print_level"] = 0  # No printing from IPOPT

        self.solver('ipopt', {
            **default_options,
            **options,
        })

        # Set the callback
        if callback is not None:
            self.callback(callback)

        # Do the actual solve
        if behavior_on_failure == "raise":
            sol = OptiSol(
                opti=self,
                cas_optisol=super().solve()
            )
        elif behavior_on_failure == "return_last":
            try:
                sol = OptiSol(
                    opti=self,
                    cas_optisol=super().solve()
                )
            except RuntimeError:
                import warnings
                warnings.warn("Optimization failed. Returning last solution.")

                sol = OptiSol(
                    opti=self,
                    cas_optisol=self.debug
                )

        if self.save_to_cache_on_solve:
            self.save_solution()

        return sol

    def solve_sweep(self,
                    parameter_mapping: Dict[cas.MX, np.ndarray],
                    update_initial_guesses_between_solves=False,
                    verbose=True,
                    solve_kwargs: Dict = None,
                    return_callable: bool = False,
                    garbage_collect_between_runs: bool = False,
                    ) -> Union[np.ndarray, Callable[[cas.MX], np.ndarray]]:

        # Handle defaults
        if solve_kwargs is None:
            solve_kwargs = {}
        solve_kwargs = {
            **dict(
                verbose=False,
                max_iter=200,
            ),
            **solve_kwargs
        }

        # Split parameter_mappings up so that it can be passed into run() via np.vectorize
        keys: Tuple[cas.MX] = tuple(parameter_mapping.keys())
        values: Tuple[np.ndarray[float]] = tuple(parameter_mapping.values())

        # Display an output
        if verbose:
            print("Running optimization sweep in serial...")

        n_runs = np.broadcast(*values).size
        run_number = 1

        def run(*args: Tuple[float]) -> Optional["OptiSol"]:
            # Collect garbage before each run, to avoid memory issues.
            if garbage_collect_between_runs:
                import gc
                gc.collect()

            # Reconstruct parameter mapping on a run-by-run basis by zipping together keys and this run's values.
            parameter_mappings_for_this_run: [cas.MX, float] = {
                k: v
                for k, v in zip(keys, args)
            }

            # Pull in run_number so that we can increment this counter
            nonlocal run_number

            # Display as needed
            if verbose:
                print(
                    "|".join(
                        [
                            f"Run {run_number}/{n_runs}".ljust(12)
                        ] + [
                            f"{v:10.5g}"
                            for v in args
                        ] + [""]
                    ),
                    end=''  # Leave the newline off, since we'll complete the line later with a success or fail print.
                )

            run_number += 1

            import time
            start_time = time.time()

            try:
                sol = self.solve(
                    parameter_mapping=parameter_mappings_for_this_run,
                    **solve_kwargs
                )

                if update_initial_guesses_between_solves:
                    self.set_initial_from_sol(sol)

                if verbose:
                    stats = sol.stats()
                    print(f" Solved in {stats['iter_count']} iterations, {time.time() - start_time:.2f} sec.")

                return sol

            except RuntimeError:
                if verbose:
                    sol = OptiSol(opti=self, cas_optisol=self.debug)
                    stats = sol.stats()
                    print(f" Failed in {stats['iter_count']} iterations, {time.time() - start_time:.2f} sec.")

                return None

        run_vectorized = np.vectorize(
            run,
            otypes='O'  # object output
        )

        sols = run_vectorized(*values)

        if return_callable:

            def get_vals(x: cas.MX) -> np.ndarray:
                return np.vectorize(
                    lambda sol: sol.value(x) if sol is not None else np.nan
                )(sols)

            return get_vals

        else:
            return sols

    ### Debugging Methods
    def find_variable_declaration(self,
                                  index: int,
                                  use_full_filename: bool = False,
                                  return_string: bool = False,
                                  ) -> Union[None, str]:
        ### Check inputs
        if index < 0:
            raise ValueError("Indices must be nonnegative.")
        if index >= self._variable_index_counter:
            raise ValueError(
                f"The variable index exceeds the number of declared variables ({self._variable_index_counter})!"
            )

        index_of_first_element = self._variable_declarations.iloc[self._variable_declarations.bisect_right(index) - 1]

        filename, lineno, code_context, n_vars = self._variable_declarations[index_of_first_element]
        source = inspect_tools.get_source_code_from_location(
            filename=filename,
            lineno=lineno,
            code_context=code_context,
        ).strip("\n")
        is_scalar = n_vars == 1
        title = f"{'Scalar' if is_scalar else 'Vector'} variable"
        if not is_scalar:
            title += f" (index {index - index_of_first_element} of {n_vars})"
        string = "\n".join([
            "",
            f"{title} defined in `{str(filename) if use_full_filename else filename.name}`, line {lineno}:",
            "",
            "```",
            source,
            "```"
        ])

        if return_string:
            return string
        else:
            print(string)

    def find_constraint_declaration(self,
                                    index: int,
                                    use_full_filename: bool = False,
                                    return_string: bool = False
                                    ) -> Union[None, str]:
        ### Check inputs
        if index < 0:
            raise ValueError("Indices must be nonnegative.")
        if index >= self._constraint_index_counter:
            raise ValueError(
                f"The constraint index exceeds the number of declared constraints ({self._constraint_index_counter})!"
            )

        index_of_first_element = self._constraint_declarations.iloc[
            self._constraint_declarations.bisect_right(index) - 1
            ]

        filename, lineno, code_context, n_cons = self._constraint_declarations[index_of_first_element]
        source = inspect_tools.get_source_code_from_location(
            filename=filename,
            lineno=lineno,
            code_context=code_context,
        ).strip("\n")
        is_scalar = n_cons == 1
        title = f"{'Scalar' if is_scalar else 'Vector'} constraint"
        if not is_scalar:
            title += f" (index {index - index_of_first_element} of {n_cons})"
        string = "\n".join([
            "",
            f"{title} defined in `{str(filename) if use_full_filename else filename.name}`, line {lineno}:",
            "",
            "```",
            source,
            "```"
        ])

        if return_string:
            return string
        else:
            print(string)

    ### Advanced Methods

    def set_initial_from_sol(self,
                             sol: cas.OptiSol,
                             initialize_primals=True,
                             initialize_duals=True,
                             ) -> None:
        """
        Sets the initial value of all variables in the Opti object to the solution of another Opti instance. Useful
        for warm-starting an Opti instance based on the result of another instance.

        Args: sol: Takes in the solution object. Assumes that sol corresponds to exactly the same optimization
        problem as this Opti instance, perhaps with different parameter values.

        Returns: None (in-place)

        """
        if initialize_primals:
            self.set_initial(self.x, sol.value(self.x))
        if initialize_duals:
            self.set_initial(self.lam_g, sol.value(self.lam_g))

    def save_solution(self):
        if self.cache_filename is None:
            raise ValueError("""In order to use the save feature, you need to supply a filepath for the cache upon
                   initialization of this instance of the Opti stack. For example: Opti(cache_filename = "cache.json")""")

        # Write a function that tries to turn an iterable into a JSON-serializable list
        def try_to_put_in_list(iterable):
            try:
                return list(iterable)
            except TypeError:
                return iterable

        # Build up a dictionary of all the variables
        solution_dict = {}
        for category, category_variables in self.variables_categorized.items():
            category_values = [
                try_to_put_in_list(self.value(variable))
                for variable in category_variables
            ]

            solution_dict[category] = category_values

        # Write the dictionary to file
        with open(self.cache_filename, "w+") as f:
            json.dump(
                solution_dict,
                fp=f,
                indent=4
            )

        return solution_dict

    def get_solution_dict_from_cache(self):
        if self.cache_filename is None:
            raise ValueError("""In order to use the load feature, you need to supply a filepath for the cache upon
                   initialization of this instance of the Opti stack. For example: Opti(cache_filename = "cache.json")""")

        with open(self.cache_filename, "r") as f:
            solution_dict = json.load(fp=f)

        # Turn all vectorized variables back into NumPy arrays
        for category in solution_dict:
            for i, var in enumerate(solution_dict[category]):
                solution_dict[category][i] = np.array(var)

        return solution_dict

    ### Methods for Dynamics and Control Problems

    def derivative_of(self,
                      variable: cas.MX,
                      with_respect_to: Union[np.ndarray, cas.MX],
                      derivative_init_guess: Union[float, np.ndarray],  # TODO add default
                      derivative_scale: Union[float, np.ndarray] = None,
                      method: str = "trapezoidal",
                      explicit: bool = False,  # TODO implement explicit
                      _stacklevel: int = 1,
                      ) -> cas.MX:
        """
        Returns a quantity that is either defined or constrained to be a derivative of an existing variable.

        For example:

        >>> opti = Opti()
        >>> position = opti.variable(init_guess=0, n_vars=100)
        >>> time = np.linspace(0, 1, 100)
        >>> velocity = opti.derivative_of(position, with_respect_to=time)
        >>> acceleration = opti.derivative_of(velocity, with_respect_to=time)

        Args:

            variable: The variable or quantity that you are taking the derivative of. The "numerator" of the
            derivative, in colloquial parlance.

            with_respect_to: The variable or quantity that you are taking the derivative with respect to. The
            "denominator" of the derivative, in colloquial parlance.

                In a typical example case, this `with_respect_to` parameter would be time. Please make sure that the
                value of this parameter is monotonically increasing, otherwise you may get nonsensical answers.

            derivative_init_guess: Initial guess for the value of the derivative. Should be either a float (in which
            case the initial guess will be a vector equal to this value) or a vector of initial guesses with the same
            length as `variable`. For more info, look at the docstring of opti.variable()'s `init_guess` parameter.

            derivative_scale: Scale factor for the value of the derivative. For more info, look at the docstring of
            opti.variable()'s `scale` parameter.

            method: The type of integrator to use to define this derivative. Options are:

                * "forward euler" - a first-order-accurate forward Euler method

                    Citation: https://en.wikipedia.org/wiki/Euler_method

                * "backwards euler" - a first-order-accurate backwards Euler method

                    Citation: https://en.wikipedia.org/wiki/Backward_Euler_method

                * "midpoint" or "trapezoid" - a second-order-accurate midpoint method

                    Citation: https://en.wikipedia.org/wiki/Midpoint_method

                * "simpson" - Simpson's rule for integration

                    Citation: https://en.wikipedia.org/wiki/Simpson%27s_rule

                * "runge-kutta" or "rk4" - a fourth-order-accurate Runge-Kutta method. I suppose that technically,
                "forward euler", "backward euler", and "midpoint" are all (lower-order) Runge-Kutta methods...

                    Citation: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#The_Runge%E2%80%93Kutta_method

                * "runge-kutta-3/8" - A modified version of the Runge-Kutta 4 proposed by Kutta in 1901. Also
                fourth-order-accurate, but all of the error coefficients are smaller than they are in the standard
                Runge-Kutta 4 method. The downside is that more floating point operations are required per timestep,
                as the Butcher tableau is more dense (i.e. not banded).

                    Citation: Kutta, Martin (1901), "Beitrag zur näherungsweisen Integration totaler
                    Differentialgleichungen", Zeitschrift für Mathematik und Physik, 46: 435–453

            explicit: If true, returns an explicit derivative rather than an implicit one. In other words,
            this *defines* the output to be a derivative of the input rather than *constraining* the output to the a
            derivative of the input.

                Explicit derivatives result in smaller, denser systems of equations that are more akin to
                shooting-type methods. Implicit derivatives result in larger, sparser systems of equations that are
                more akin to collocation methods. Explicit derivatives are better for simple, stable systems with few
                states, while implicit derivatives are better for complex, potentially-unstable systems with many
                states.

                # TODO implement explicit

            _stacklevel: Optional and advanced, purely used for debugging. Allows users to correctly track where
            constraints are declared in the event that they are subclassing `aerosandbox.Opti`. Modifies the
            stacklevel of the declaration tracked, which is then presented using
            `aerosandbox.Opti.variable_declaration()` and `aerosandbox.Opti.constraint_declaration()`.


        Returns: A vector consisting of the derivative of the parameter `variable` with respect to `with_respect_to`.

        """
        ### Set defaults
        # if with_respect_to is None:
        #     with_respect_to = np.ones(shape=np.length(variable)) # TODO consider whether we want to even allow this...
        # if derivative_init_guess is None:
        #     raise NotImplementedError() # TODO implement default value for this

        ### Check inputs
        N = np.length(variable)
        if not np.length(with_respect_to) == N:
            raise ValueError("The inputs `variable` and `with_respect_to` must be vectors of the same length!")

        ### Clean inputs
        method = method.lower()

        ### Implement the derivative
        if not explicit:
            derivative = self.variable(
                init_guess=derivative_init_guess,
                n_vars=N,
                scale=derivative_scale,
            )

            self.constrain_derivative(
                derivative=(
                    derivative / derivative_scale
                    if derivative_scale is not None
                    else derivative
                ),
                variable=(
                    variable / derivative_scale
                    if derivative_scale is not None
                    else variable
                ),
                with_respect_to=with_respect_to,
                method=method,
                _stacklevel=_stacklevel + 1
            )

        else:
            raise NotImplementedError("Haven't yet implemented explicit derivatives! Use implicit ones for now...")

        return derivative

    def constrain_derivative(self,
                             derivative: cas.MX,
                             variable: cas.MX,
                             with_respect_to: Union[np.ndarray, cas.MX],
                             method: str = "trapezoidal",
                             _stacklevel: int = 1,
                             ) -> None:
        """
        Adds a constraint to the optimization problem such that:

            d(variable) / d(with_respect_to) == derivative

        Can be used directly; also called indirectly by opti.derivative_of() for implicit derivative creation.

        Args:
            derivative: The derivative that is to be constrained here.

            variable: The variable or quantity that you are taking the derivative of. The "numerator" of the
            derivative, in colloquial parlance.

            with_respect_to: The variable or quantity that you are taking the derivative with respect to. The
            "denominator" of the derivative, in colloquial parlance.

                In a typical example case, this `with_respect_to` parameter would be time. Please make sure that the
                value of this parameter is monotonically increasing, otherwise you may get nonsensical answers.

            method: The type of integrator to use to define this derivative. Options are:

                * "forward euler" - a first-order-accurate forward Euler method

                    Citation: https://en.wikipedia.org/wiki/Euler_method

                * "backwards euler" - a first-order-accurate backwards Euler method

                    Citation: https://en.wikipedia.org/wiki/Backward_Euler_method

                * "midpoint" or "trapezoid" - a second-order-accurate midpoint method

                    Citation: https://en.wikipedia.org/wiki/Midpoint_method

                * "simpson" - Simpson's rule for integration

                    Citation: https://en.wikipedia.org/wiki/Simpson%27s_rule

                * "runge-kutta" or "rk4" - a fourth-order-accurate Runge-Kutta method. I suppose that technically,
                "forward euler", "backward euler", and "midpoint" are all (lower-order) Runge-Kutta methods...

                    Citation: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#The_Runge%E2%80%93Kutta_method

                * "runge-kutta-3/8" - A modified version of the Runge-Kutta 4 proposed by Kutta in 1901. Also
                fourth-order-accurate, but all of the error coefficients are smaller than they are in the standard
                Runge-Kutta 4 method. The downside is that more floating point operations are required per timestep,
                as the Butcher tableau is more dense (i.e. not banded).

                    Citation: Kutta, Martin (1901), "Beitrag zur näherungsweisen Integration totaler
                    Differentialgleichungen", Zeitschrift für Mathematik und Physik, 46: 435–453

            Note that all methods are expressed as integrators rather than differentiators; this prevents
            singularities from forming in the limit of timestep approaching zero. (For those coming from the PDE
            world, this is analogous to using finite volume methods rather than finite difference methods to allow
            shock capturing.)

            _stacklevel: Optional and advanced, purely used for debugging. Allows users to correctly track where
            constraints are declared in the event that they are subclassing `aerosandbox.Opti`. Modifies the
            stacklevel of the declaration tracked, which is then presented using
            `aerosandbox.Opti.variable_declaration()` and `aerosandbox.Opti.constraint_declaration()`.

        Returns: None (adds constraint in-place).

        """
        try:
            derivative[0]
        except (TypeError, IndexError):
            derivative = np.ones_like(with_respect_to) * derivative

        # TODO scale constraints by variable scale?
        # TODO make
        from aerosandbox.numpy.integrate_discrete import integrate_discrete_intervals

        integrals = integrate_discrete_intervals(
            f = derivative,
            x = with_respect_to,
            multiply_by_dx=True,
            method=method
        )
        duals = self.subject_to(
            np.diff(variable) == integrals,
            _stacklevel=_stacklevel + 1
        )

        return duals

class OptiSol:
    def __init__(self,
                 opti: Opti,
                 cas_optisol: cas.OptiSol
                 ):
        """
        An OptiSol object represents a solution to an optimization problem. This class is a wrapper around CasADi's
        `OptiSol` class that provides convenient solution query utilities for various Python data types.

        Args:
            opti: The `Opti` object that generated this solution.

            cas_optisol: The `casadi.OptiSol` object from CasADi's optimization solver.

        Returns:
            An `OptiSol` object.

        Usage:
            >>> # Initialize an Opti object.
            >>> opti = asb.Opti()
            >>>
            >>> # Define a scalar variable.
            >>> x = opti.variable(init_guess=2.0)
            >>>
            >>> # Define an objective function.
            >>> opti.minimize(x ** 2)
            >>>
            >>> # Solve the optimization problem. `sol` is now a
            >>> sol = opti.solve()
            >>>
            >>> # Retrieve the value of the variable x in the solution:
            >>> x_value = sol.value(x)
            >>>
            >>> # Or, to be more concise:
            >>> x_value = sol(x)
        """
        self.opti = opti
        self._sol = cas_optisol

    def __call__(self, x: Union[cas.MX, np.ndarray, float, int, List, Tuple, Set, Dict, Any]) -> Any:
        """
        A shorthand alias for `sol.value(x)`. See `OptiSol.value()` documentation for details.

        Args:
            x: A Python data structure to substitute values into, using the solution in this OptiSol object.

        Returns:

            A copy of `x`, where all symbolic optimization variables (recursively substituted at unlimited depth)
            have been converted to float or array values.

        """
        return self.value(x)

    def _value_scalar(self, x: Union[cas.MX, np.ndarray, float, int]) -> Union[float, np.ndarray]:
        """
        Gets the value of a variable at the solution point. For developer use - see following paragraph.

        This method is basically a less-powerful version of calling either `sol(x)` or `sol.value(x)` - if you're a
            user and not a developer, you almost-certainly want to use one of those methods instead, as those are less
            fragile with respect to various input data types. This method exists only as an abstraction to make it easier
            for other developers to subclass OptiSol, if they wish to intercept the variable substitution process.

        Args:
            x:

        Returns:

        """
        return self._sol.value(x)

    def value(self,
              x: Union[cas.MX, np.ndarray, float, int, List, Tuple, Set, Dict, Any],
              recursive: bool = True,
              warn_on_unknown_types: bool = False
              ) -> Any:
        """
        Gets the value of a variable (or a data structure) at the solution point. This solution point is the optimum,
            if the optimization process solved successfully.

        On a computer science level, this method converts a symbolic optimization variable to a concrete float or
            array value. More generally, it converts any Python data structure (along with any of its contents,
            recursively, at unlimited depth), replacing any symbolic optimization variables it finds with concrete float
            or array values.

        Note that, for convenience, you can simply call:
        >>> sol(x)
        if you prefer. This is equivalent to calling this method with the syntax:
        >>> sol.value(x)
        (these are aliases of each other)

        Args:
            x: A Python data structure to substitute values into, using the solution in this OptiSol object.

            recursive: If True, the substitution will be performed recursively. Otherwise, only the top-level data
                structure will be converted.

            warn_on_unknown_types: If True, a warning will be issued if a data type that cannot be converted or
                parsed as definitively un-convertable is encountered.

        Returns:
            A copy of `x`, where all symbolic optimization variables (recursively substituted at unlimited depth)
                have been converted to float or array values.

        Usage:



        """
        if not recursive:
            return self._value_scalar(x)

        # If it's a CasADi type, do the conversion, and call it a day.
        if np.is_casadi_type(x, recursive=False):
            return self._value_scalar(x)

        t = type(x)

        # If it's a Python iterable, recursively convert it, and preserve the type as best as possible.
        if issubclass(t, list):
            return [self.value(i) for i in x]
        if issubclass(t, tuple):
            return tuple([self.value(i) for i in x])
        if issubclass(t, (set, frozenset)):
            return {self.value(i) for i in x}
        if issubclass(t, dict):
            return {
                self.value(k): self.value(v)
                for k, v in x.items()
            }

        # Skip certain Python types
        if issubclass(t, (
                bool, str,
                int, float, complex,
                range,
                type(None),
                bytes, bytearray, memoryview,
                type,
        )):
            return x

        # Skip certain CasADi types
        if issubclass(t, (
                cas.Opti, cas.OptiSol,
        )):
            return x

        # If it's any other type, try converting its attribute dictionary, if it has one:
        try:
            new_x = copy.copy(x)

            for k, v in x.__dict__.items():
                setattr(new_x, k, self.value(v))

            return new_x

        except (AttributeError, TypeError):
            pass

        # Try converting it blindly. This will catch most NumPy-array-like types.
        try:
            return self._value_scalar(x)
        except (NotImplementedError, TypeError, ValueError):
            pass

        # At this point, we're not really sure what type the object is. Raise a warning if directed and return the
        # item, then hope for the best.
        if warn_on_unknown_types:
            import warnings
            warnings.warn(f"In solution substitution, could not convert an object of type {t}.\n"
                          f"Returning it and hoping for the best.", UserWarning)

        return x

    def stats(self) -> Dict[str, Any]:
        return self._sol.stats()

    def value_variables(self):
        return self._sol.value_variables()

    def value_parameters(self):
        return self._sol.value_parameters()

    def show_infeasibilities(self, tol: float = 1e-3) -> None:
        """
        Prints a summary of any violated constraints in the solution.

        Args:

            tol: The tolerance for violation. If the constraint is violated by less than this amount, it will not be
                printed.

        Returns: None (prints to console)
        """
        lbg = self(self.opti.lbg)
        ubg = self(self.opti.ubg)

        g = self(self.opti.g)

        constraint_violated = np.logical_or(
            g + tol < lbg,
            g - tol > ubg
        )

        lbg_isfinite = np.isfinite(lbg)
        ubg_isfinite = np.isfinite(ubg)

        for i in np.arange(len(g)):
            if constraint_violated[i]:
                print("-" * 50)

                if lbg_isfinite[i] and ubg_isfinite[i]:
                    if lbg[i] == ubg[i]:
                        print(f"{lbg[i]} == {g[i]} (violation: {np.abs(g[i] - lbg[i])})")
                    else:
                        print(f"{lbg[i]} < {g[i]} < {ubg[i]} (violation: {np.maximum(lbg[i] - g[i], g[i] - ubg[i])})")
                elif lbg_isfinite[i] and not ubg_isfinite[i]:
                    print(f"{lbg[i]} < {g[i]} (violation: {lbg[i] - g[i]})")
                elif not lbg_isfinite[i] and ubg_isfinite[i]:
                    print(f"{g[i]} < {ubg[i]} (violation: {g[i] - ubg[i]})")
                else:
                    raise ValueError(
                        "Contact the AeroSandbox developers if you see this message; it should be impossible.")

                self.opti.find_constraint_declaration(index=i)


if __name__ == '__main__':
    import pytest

    # pytest.main()

    opti = Opti()  # set up an optimization environment

    a = opti.parameter(1)
    b = opti.parameter(100)

    # Define optimization variables
    x = opti.variable(init_guess=0)
    y = opti.variable(init_guess=0)

    # Define objective
    f = (a - x) ** 2 + b * (y - x ** 2) ** 2
    opti.minimize(f)

    opti.subject_to([
        x ** 2 + y ** 2 <= 1
    ])

    # Optimize
    sol = opti.solve()

    assert sol([x, y]) == pytest.approx([0.7864, 0.6177], abs=1e-3)
