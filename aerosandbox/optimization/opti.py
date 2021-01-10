import casadi as cas
from typing import Union, List, Dict
import numpy as np
import pytest
import json

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
                 n_vars: int = 1,
                 init_guess: Union[float, np.ndarray] = None,
                 scale: float = None,
                 log_transform: bool = False,
                 category: str = "Uncategorized",
                 freeze: bool = False,
                 ) -> cas.MX:
        """
        Initializes a new decision variable (or vector of decision variables, if n_vars != 1).
        
        It is recommended that you provide an initial guess (`init_guess`) and scale (`scale`) for each variable,
        although these are not strictly required.

        Args:
            n_vars: [Optional] Number of variables to initialize (used to initialize a vector of variables). If you are
                initializing a scalar variable (the most typical case), leave this equal to 1. When using vector variables,
                individual components of this vector of variables can be accessed via normal indexing.

                Example:
                    >>> opti = asb.Opti()
                    >>> my_var = opti.variable(n_vars = 5)
                    >>> opti.subject_to(my_var[3] >= my_var[2])  # This is a valid way of indexing
                    >>> my_sum = asb.cas.sum1(my_var)  # This will sum up all elements of `my_var`

            init_guess: [Optional] Initial guess for the variable being initialized. For scalar variables, this should be a
                float. For vector variables (see `n_vars`), you can provide either a float (in which case all elements
                of the vector will be initialized to the given value) or an iterable of equal length (in which case
                each element will be initialized to the corresponding value in the given iterable).

                In the case where the variable is to be log-transformed (see `log_transform`), the initial guess should
                not be log-transformed as well; this happens under the hood. The initial guess must, of course, be a
                positive number in this case.

                If not specified, initial guess defaults to 0 for non-log-transformed variables and 1 for
                log-transformed variables.

            scale: [Optional] Approximate scale of the variable. If not specified, defaults to the supplied initial
                guess if one exists; otherwise, defaults to 1.

            log_transform: [Optional] Advanced use only. A flag of whether to internally-log-transform this variable
            before passing it to the optimizer. Good for known positive engineering quantities that become nonsensical
            if negative (e.g. mass). Log-transforming these variables can help maintain convexity.

            category: [Optional] What category of variables does this belong to

        Returns:
            The variable itself as a symbolic CasADi variable (MX type).

        """
        # Validate the inputs
        if log_transform and init_guess is not None:
            if np.any(init_guess <= 0):
                raise ValueError(
                    "If you are initializing a log-transformed variable, the initial guess(es) must be positive.")

        # Set defaults
        if init_guess is None:
            init_guess = 1 if log_transform else 0
        if scale is None:
            scale = init_guess if log_transform else 1

        # Validate the inputs
        if np.any(scale <= 0):
            raise ValueError("The 'scale' argument must be a positive number.")

        # If the variable is in a category to be frozen, fix the variable at the initial guess.
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


        Returns: An OptiSol object that contains the solved optimization problem. To extract values, use
            OptiSol.value(variable).

            Example:
                >>> sol = opti.solve()
                >>> x_star = sol.value(x) # Get the value of variable x at the optimum.

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

                new_parameter_mappings = {
                    k: v for k, v in zip(category_variables, category_values)
                }
                parameter_mapping = {**parameter_mapping, **new_parameter_mappings}

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

        # Do the actual solve
        sol = super().solve()

        if self.save_to_cache_on_solve:
            self.save_solution()

        return sol


if __name__ == '__main__':
    pytest.main()
