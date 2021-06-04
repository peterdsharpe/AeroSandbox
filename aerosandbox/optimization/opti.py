from typing import Union, List, Dict, Callable
import json
import casadi as cas
import aerosandbox.numpy as np


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
                 variable_categories_to_freeze: List[str] = [],
                 cache_filename: str = None,
                 load_frozen_variables_from_cache: bool = False,
                 save_to_cache_on_solve: bool = False,
                 ignore_violated_parametric_constraints: bool = False,
                 ):

        # Parent class initialization
        super().__init__()

        # Initialize class variables
        self.variable_categories_to_freeze = variable_categories_to_freeze
        self.cache_filename = cache_filename
        self.load_frozen_variables_from_cache = load_frozen_variables_from_cache  # TODO load and start tracking
        self.save_to_cache_on_solve = save_to_cache_on_solve
        self.ignore_violated_parametric_constraints = ignore_violated_parametric_constraints

        # Start tracking variables and categorize them.
        self.variables_categorized = {}  # key: value :: category name [str] : list of variables [list]

    def variable(self,
                 init_guess: Union[float, np.ndarray],
                 n_vars: int = None,
                 scale: float = None,
                 freeze: bool = False,
                 log_transform: bool = False,
                 category: str = "Uncategorized",
                 lower_bound: float = None,
                 upper_bound: float = None,
                 ) -> cas.MX:
        """
        Initializes a new decision variable (or vector of decision variables). You must pass an initial guess (
        `init_guess`) upon defining a new variable. Dimensionality is inferred from this initial guess, but it can be
        overridden; see below for syntax.

        It is highly, highly recommended that you provide a scale (`scale`) for each variable, especially for
        nonconvex problems, although this is not strictly required.

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

            category: [Optional] What category of variables does this belong to?

        Usage notes:

            When using vector variables, individual components of this vector of variables can be accessed via normal
            indexing. Example:
                >>> opti = asb.Opti()
                >>> my_var = opti.variable(n_vars = 5)
                >>> opti.subject_to(my_var[3] >= my_var[2])  # This is a valid way of indexing
                >>> my_sum = asb.sum(my_var)  # This will sum up all elements of `my_var`

        Returns:
            The variable itself as a symbolic CasADi variable (MX type).

        """
        ### Set defaults
        if n_vars is None:  # Infer dimensionality from init_guess if it is not provided
            n_vars = np.length(init_guess)
        if scale is None:  # Infer a scale from init_guess if it is not provided
            if log_transform:
                scale = 1
            else:
                scale = np.mean(np.fabs(init_guess))  # Initialize the scale to a heuristic based on the init_guess
                if scale == 0:  # If that heuristic leads to a scale of 0, use a scale of 1 instead.
                    scale = 1

                # scale = np.fabs(
                #     np.where(
                #         init_guess != 0,
                #         init_guess,
                #         1
                #     ))

        # Validate the inputs
        if log_transform:
            if np.any(init_guess <= 0):
                raise ValueError(
                    "If you are initializing a log-transformed variable, the initial guess(es) must all be positive.")
        if np.any(scale <= 0):
            raise ValueError("The 'scale' argument must be a positive number.")

        # If the variable is in a category to be frozen, fix the variable at the initial guess.
        is_manually_frozen = freeze
        if category in self.variable_categories_to_freeze:
            freeze = True

        # If the variable is to be frozen, return the initial guess. Otherwise, define the variable using CasADi symbolics.
        if freeze:
            var = self.parameter(n_params=n_vars, value=init_guess)
        else:
            if not log_transform:
                var = scale * super().variable(n_vars)
                self.set_initial(var, init_guess)
            else:
                log_scale = scale / init_guess
                log_var = log_scale * super().variable(n_vars)
                var = np.exp(log_var)
                self.set_initial(log_var, np.log(init_guess))

        # Track the variable
        if category not in self.variables_categorized:  # Add a category if it does not exist
            self.variables_categorized[category] = []
        self.variables_categorized[category].append(var)
        var.is_manually_frozen = is_manually_frozen

        # Apply bounds
        if lower_bound is not None:
            self.subject_to(var >= lower_bound)
        if upper_bound is not None:
            self.subject_to(var <= upper_bound)

        return var

    def subject_to(self,
                   constraint: Union[cas.MX, bool, List],  # TODO add scale
                   ) -> cas.MX:
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

        Returns: The dual variable associated with the new constraint. If the `constraint` input is a list, returns
            a list of dual variables.

        """
        # Determine whether you're dealing with a single (possibly vectorized) constraint or a list of constraints.
        # If the latter, recursively apply them.
        if type(constraint) in (list, tuple):
            return [
                self.subject_to(each_constraint)  # return the dual of each constraint
                for each_constraint in constraint
            ]

        # If it's a proper constraint (MX-type and non-parametric),
        # pass it into the parent class Opti formulation and be done with it.
        if isinstance(constraint, cas.MX) and not self.advanced.is_parametric(constraint):
            super().subject_to(constraint)
            dual = self.dual(constraint)

            return dual
        else:  # Constraint is not valid because it is not MX type or is parametric.
            try:
                constraint_satisfied = np.all(self.value(constraint))
            except:
                raise TypeError(f"""Opti.subject_to could not determine the truthiness of your constraint, and it
                    doesn't appear to be a symbolic type or a boolean type. You supplied the following constraint:
                    {constraint}""")

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

    def parameter(self,
                  value: Union[float, np.ndarray] = 0.,
                  n_params: int = None,
                  ) -> cas.MX:
        """
        Initializes a new parameter (or vector of parameters). You must pass a value (`value`) upon defining a new
        parameter. Dimensionality is inferred from this valXPue, but it can be overridden; see below for syntax.

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

            n_vars: [Optional] Used to manually override the dimensionality of the parameter to create; if not
            provided, the dimensionality of the parameter is inferred from `value`.

                The only real case where you need to use this argument would be if you are initializing a vector
                parameter to a scalar value, but you don't feel like using `value=my_value * np.ones(n_vars)`.
                For example:

                    >>> opti = asb.Opti()
                    >>> vector_var = opti.parameter(value=5, n_vars=10) # Initializes a vector parameter of length
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

    def derivative_of(self,
                      variable: cas.MX,
                      with_respect_to: Union[np.ndarray, cas.MX],
                      derivative_init_guess: Union[float, np.ndarray],  # TODO add default
                      derivative_scale: float = None,
                      method: str = "midpoint",
                      regularize: bool = False,
                      explicit: bool = False  # TODO implement explicit
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

            regularize: Most of these integration methods result in N-1 constraints for a problem with N state
            variables. This makes the problem ill-posed, as there is an extra degree of freedom added to the problem.
            If the regularize flag is set True, we will automatically add one more constraint to make the problem
            well-posed. The specific constraint that is added depends on the integration method used.

            explicit: If true, returns an explicit derivative rather than an implicit one. In other words,
            this *defines* the output to be a derivative of the input rather than *constraining* the output to the a
            derivative of the input.

                Explicit derivatives result in smaller, denser systems of equations that are more akin to
                shooting-type methods. Implicit derivatives result in larger, sparser systems of equations that are
                more akin to collocation methods. Explicit derivatives are better for simple, stable systems with few
                states, while implicit derivatives are better for complex, potentially-unstable systems with many
                states.

                # TODO implement explicit



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
                derivative=derivative,
                variable=variable,
                with_respect_to=with_respect_to,
                method=method,
                regularize=regularize
            )

        else:
            raise NotImplementedError("Haven't yet implemented explicit derivatives! Use implicit ones for now...")

        return derivative

    def constrain_derivative(self,
                             derivative: cas.MX,
                             variable: cas.MX,
                             with_respect_to: Union[np.ndarray, cas.MX],
                             method: str = "midpoint",
                             regularize: bool = False,
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

            regularize: Most of these integration methods result in N-1 constraints for a problem with N state
            variables. This makes the problem ill-posed, as there is an extra degree of freedom added to the problem.
            If the regularize flag is set True, we will automatically add one more constraint to make the problem
            well-posed. The specific constraint that is added depends on the integration method used.

        Returns: None (adds constraint in-place).

        """
        d_var = np.diff(variable)
        d_time = np.diff(with_respect_to)  # Calculate the timestep

        # TODO scale constraints by variable scale?
        # TODO make

        if method == "forward euler":
            raise NotImplementedError
            self.subject_to(
                d_var == derivative[:-1] * d_time
            )
            self.subject_to(
                derivative[-1] == derivative[-2]  # First-order constraint at last point
            )


        elif method == "backward euler":
            raise NotImplementedError
            self.subject_to(
                d_var == derivative[1:] * d_time
            )
            self.subject_to(
                derivative[0] == derivative[1]  # First-order constraint at first point
            )

        elif method == "midpoint" or method == "trapezoid":
            self.subject_to(
                d_var == np.trapz(derivative) * d_time,
            )
            if regularize:
                # Apply a second-order constraint at the first point
                coefficients = np.finite_difference_coefficients(
                    x=with_respect_to[:3],
                    x0=with_respect_to[0],
                    derivative_degree=1
                )
                derivative_value = np.sum(variable[:3] * coefficients)
                self.subject_to(
                    derivative[0] == derivative_value
                )


        elif method == "simpson":
            raise NotImplementedError

        elif method == "runge-kutta" or method == "rk4":
            raise NotImplementedError

        elif method == "runge-kutta-3/8":
            raise NotImplementedError

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

    def solve(self,
              parameter_mapping: Dict[cas.MX, float] = None,
              max_iter: int = 1000,
              callback: Callable = None,
              verbose: bool = True,
              jit: bool = False,  # TODO document, add unit tests for jit
              options: Dict = None,  # TODO document
              ) -> cas.OptiSol:
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

            callback: [Optional] A function to be called at each iteration of the optimization algorithm.
                Useful for printing progress or displaying intermediate results.

                The callback function `func` should have the syntax `func(iteration_number)`, where iteration_number
                is an integer corresponding to the current iteration number. In order to access intermediate
                quantities of optimization variables (e.g. for plotting), use the `Opti.debug.value(x)` syntax for
                each variable `x`.

            verbose: Should we print the output of IPOPT?

        Returns: An OptiSol object that contains the solved optimization problem. To extract values, use
            OptiSol.value(variable).

            Example:
                >>> sol = opti.solve()
                >>> x_opt = sol.value(x) # Get the value of variable x at the optimum.

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
            size_k = np.product(k.shape)
            try:
                size_v = np.product(v.shape)
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

        if jit:
            options["jit"] = True
            # options["compiler"] = "shell"  # Recommended by CasADi devs, but doesn't work on my machine
            options["jit_options"] = {
                "flags": ["-O3"],
                # "verbose": True
            }

        options["ipopt.sb"] = 'yes'

        if verbose:
            options["ipopt.print_level"] = 5
        else:
            options["print_time"] = False
            options["ipopt.print_level"] = 0

        # Set defaults, if not set
        if "ipopt.max_iter" not in options:
            options["ipopt.max_iter"] = max_iter
        if "ipopt.mu_strategy" not in options:
            options["ipopt.mu_strategy"] = "adaptive"

        self.solver('ipopt', options)

        # Set the callback
        if callback is not None:
            self.callback(callback)

        # Do the actual solve
        sol = super().solve()

        if self.save_to_cache_on_solve:
            self.save_solution()

        return sol


if __name__ == '__main__':
    import pytest

    pytest.main()
