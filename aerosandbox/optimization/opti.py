import casadi as cas
from typing import Union, List, Dict, Callable
import numpy as np
import pytest
import json
from aerosandbox.optimization.math import *


class Opti(cas.Opti):
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
        self.load_frozen_variables_from_cache = load_frozen_variables_from_cache
        self.save_to_cache_on_solve = save_to_cache_on_solve
        self.ignore_violated_parametric_constraints = ignore_violated_parametric_constraints

        # Start tracking variables and categorize them.
        self.variables_categorized = {}  # key: value :: category name [str] : list of variables [list]

    def variable(self,
                 init_guess: Union[float, np.ndarray],
                 n_vars: int = None,
                 scale: float = None,
                 log_transform: bool = False,
                 category: str = "Uncategorized",
                 freeze: bool = False,
                 ) -> cas.MX:
        """
        Initializes a new decision variable (or vector of decision variables, if n_vars != 1).

        It is highly, highly recommended that you provide a scale (`scale`) for each variable, especially for
        nonconvex problems, although this is not strictly required.

        Args:

            init_guess: Initial guess for the optimal value of the variable being initialized. This is where in the
            design space the optimizer will start looking.

                This can be either a float or a NumPy ndarray; the dimension of the variable (i.e. scalar,
                vector) that is created will be automatically inferred from the shape of the initial guess you
                provide here. (Although it can also be overridden using the `n_vars` parameter; see below.)

                For scalar variables, your initial guess should be a float.
                >>> opti = asb.Opti()
                >>> scalar_var = opti.variable(init_guess=5) # Initializes a scalar variable at a value of 5

                For vector variables, your initial guess should be either:

                    * a float, in which case you must pass the length of the vector as `n_vars` otherwise a scalar
                    variable will be created
                    >>> opti = asb.Opti()
                    >>> vector_var = opti.variable(init_guess=5, n_vars=10) # Initializes a vector variable of length
                        # 10, with all 10 elements set to an initial guess of 5.

                    * a NumPy ndarray, in which case each element will be initialized to the corresponding value in
                    the given array.
                    >>> opti = asb.Opti()
                    >>> vector_var = opti.variable(init_guess=np.linspace(0, 5, 10)) # Initializes a vector variable of
                        # length 10, with all 10 elements initialized to linearly vary between 0 and 5.

                In the case where the variable is to be log-transformed (see `log_transform`), the initial guess
                should not be log-transformed as well - just supply the initial guess as usual. (Log-transform of the
                initial guess happens under the hood.) The initial guess must, of course, be a positive number in
                this case.

            n_vars: [Optional] Used to manually override the dimensionality of the variable to create; if not
            provided, the dimensionality of the variable is inferred from the initial guess `init_guess`.

                The only real case where you need to use this argument would be if you are initializing a vector
                variable to a scalar value; for example:
                    >>> opti = asb.Opti()
                    >>> vector_var = opti.variable(init_guess=5, n_vars=10) # Initializes a vector variable of length
                        # 10, with all 10 elements set to an initial guess of 5.


            scale: [Optional] Approximate scale of the variable.

                For example, if you're optimizing the design of a car and setting the wheel diameter as an
                optimization variable, you might choose `scale=0.5`, corresponding to 0.5 meters.

                Properly scaling your variables can have a huge impact on solution speed (or even if the optimizer
                converges at all). Although the optimizer IPOPT is theoretically scale-invariant, numerical precision
                issues due to floating-point arithmetic can make solving poorly-scaled problems really difficult. See
                here for more info: https://web.casadi.org/blog/nlp-scaling/

                If not specified, the code will try to pick a sensible value by defaulting to the `init_guess`.



            log_transform: [Optional] Advanced use only. A flag of whether to internally-log-transform this variable
            before passing it to the optimizer. Good for known positive engineering quantities that become nonsensical
            if negative (e.g. mass). Log-transforming these variables can help maintain convexity.

            category: [Optional] What category of variables does this belong to

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
        if n_vars is None: # Infer dimensionality from init_guess if it is not provided
            try:
                n_vars = len(init_guess)
            except TypeError: # init_guess has no function len() -> either float, int, or CasADi type
                try:
                    n_vars = init_guess.shape[0]
                except AttributeError: # init_guess has no attribute shape -> either float or int
                    n_vars = 1
        if scale is None: # Infer a scale from init_guess if it is not provided
            if log_transform:
                scale = 1
            else:
                scale = if_else( # Initialize the scale to the init_guess, unless it's zero, in which case use 1.
                    init_guess != 0,
                    init_guess,
                    1
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
        if category in self.variable_categories_to_freeze:
            freeze = True

        # If the variable is to be frozen, return the initial guess. Otherwise, define the variable using CasADi symbolics.
        if freeze:
            # var = init_guess * np.ones(n_vars)
            var = self.parameter(n_params=n_vars, value=init_guess)
        else:
            if not log_transform:
                var = scale * super().variable(n_vars)
                self.set_initial(var, init_guess)
            else:
                log_scale = scale / init_guess
                log_var = log_scale * super().variable(n_vars)
                var = cas.exp(log_var)
                self.set_initial(log_var, cas.log(init_guess))

        # Track the variable
        if category not in self.variables_categorized:  # Add a category if it does not exist
            self.variables_categorized[category] = []
        self.variables_categorized[category].append(var)
        var.is_manually_frozen = is_manually_frozen

        return var

    def subject_to(self,
                   constraint: Union[cas.MX, bool, List],
                   ) -> cas.MX:
        """
        Initialize a new constraint(s).

        Args:
            constraint: A constraint that you want to hold true at the optimum. Example:
                >>> x = opti.variable()
                >>> opti.subject_to(x >= 5)
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
        if isinstance(constraint, List):
            return [
                self.subject_to(each_constraint)  # return the dual of each constraint
                for each_constraint in constraint
            ]

        # If it's a proper constraint (MX type and non-parametric),
        # pass it into the problem formulation and be done with it.
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
                  n_params: int = 1,
                  value: float = 0.,
                  ) -> cas.MX:
        """
        Initialize a new parameter (or vector of paramters, if n_params != 1).

        Args:
            n_params: [Optional] Number of parameters to initialize (used to initialize a vector of parameters). If you
                are initializing a scalar parameter (the most typical case), leave this equal to 1. When using vector
                parameters, inidividual components of this vector of parameters can be aaccessed via normal indexing.

                Example:
                    >>> opti = asb.Opti()
                    >>> my_param = opti.parameter(n_params = 5)
                    >>> for i in range(5):
                    >>>     print(my_param[i]) # This is a valid way of indexing

            value: Value to set the parameter to. Defaults to zero.
                The value can alternatively be manually set (or overwritten) after parameter initialization
                using the syntax:
                >>> param = opti.parameter()
                >>> opti.set_value(param, 5)
                Which initializes a new parameter and sets its value to 5.


        Returns:
            The parameter itself as a symbolic CasADi variable (MX type).

        """
        param = super().parameter(n_params)
        self.set_value(param, value)

        return param

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
              max_iter: int = 3000,
              callback: Callable = None,
              solver: str = 'ipopt'
              ) -> cas.OptiSol:
        """
        Solve the optimization problem.

        Args:
            parameter_mapping: [Optional] Allows you to specify values for parameters.
                Dictionary where the key is the parameter and the value is the value to be set to.

                Example:
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
                is an integer corresponding to the current iteration number. In order to access intermediate quantities
                of optimization variables, use the `Opti.debug.value(x)` syntax for each variable `x`.

            solve: [Optional] Which optimization backend do you wish to use? [str] Only tested with "ipopt".

        Returns: An OptiSol object that contains the solved optimization problem. To extract values, use
            OptiSol.value(variable).

            Example:
                >>> sol = opti.solve()
                >>> x_opt = sol.value(x) # Get the value of variable x at the optimum.

        """
        if parameter_mapping is None:
            parameter_mapping = {}

        # If you're loading frozen variables from cache, do it here:
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

        # Map any parameters to needed values
        for k, v in parameter_mapping.items():
            size_k = np.product(k.shape)
            size_v = np.product(v.shape)
            if size_k != size_v:
                raise RuntimeError("""Problem with loading cached solution: it looks like the length of a vectorized 
                variable has changed since the cached solution was saved (or variables were defined in a different order). 
                Because of this, the cache cannot be loaded. 
                Re-run the original optimization study to regenerate the cached solution.""")

            self.set_value(k, v)

        # Set solver settings.
        p_opts = {}
        s_opts = {}
        s_opts["max_iter"] = max_iter
        s_opts["mu_strategy"] = "adaptive"
        self.solver(solver, p_opts, s_opts)  # Default to IPOPT solver

        # Set the callback
        if callback is not None:
            self.callback(callback)

        # Do the actual solve
        sol = super().solve()

        if self.save_to_cache_on_solve:
            self.save_solution()

        return sol


if __name__ == '__main__':
    pytest.main()
