import casadi as cas
from typing import Union, List
import numpy as np
import pytest


class Opti(cas.Opti):
    def __init__(self,
                 variable_categories_to_freeze=[],
                 ):

        # Parent class initialization
        super().__init__()

        # Set default solver settings.
        p_opts = {}
        s_opts = {}
        s_opts["max_iter"] = 3000
        s_opts["mu_strategy"] = "adaptive"
        self.solver('ipopt', p_opts, s_opts)  # Default to IPOPT solver

        # Start tracking variables, constraints (dual variables), and parameters for operations later on.
        self.variable_list = []
        self.constraints_list = []
        self.parameter_list = []

        # Start tracking variable categories
        self.variable_categories = []
        self.variable_categories_to_freeze = variable_categories_to_freeze

    def variable(self,
                 n_vars: int = 1,
                 init_guess: Union[float, np.ndarray] = None,
                 scale: float = None,
                 log_transform: bool = False,
                 category: str = "Uncategorized",
                 freeze: bool = False,
                 ) -> cas.MX:
        """
        Initializes a new decision variable.
        
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

        # Add the category name to the list to be tracked
        if category not in self.variable_categories:
            self.variable_categories.append(category)

        # If the variable is in a category to be frozen, fix the variable at the initial guess.
        if category in self.variable_categories_to_freeze:
            freeze = True

        # If the variable is to be frozen, return the initial guess. Otherwise, define the variable using CasADi symbolics.
        if freeze:
            var = init_guess * np.ones(n_vars)
        else:
            if not log_transform:
                var = scale * super().variable(n_vars)
                self.set_initial(var, init_guess)
            else:
                log_scale = scale / init_guess
                log_var = log_scale * super().variable(n_vars)
                var = cas.exp(log_var)
                self.set_initial(log_var, cas.log(init_guess))

        # Add the variable to the variable list to be tracked
        self.variable_list.append(var)

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
                self.subject_to(each_constraint) # return the dual of each constraint
                for each_constraint in constraint
            ]

        # If it's a proper constraint (MX type), pass it into the problem formulation and be done with it.
        if isinstance(constraint, cas.MX):
            super().subject_to(constraint)
            dual = self.dual(constraint)

            # Add the dual to the constraints list to be tracked
            self.constraints_list.append(dual)

            return dual

        # If the constraint(s) always evaluates True (e.g. if you enter "5 > 3"), skip it.
        # This allows you to toggle frozen variables without causing problems with setting up constraints.
        elif np.all(constraint):
            return None # dual of an always-true constraint doesn't make sense to evaluate.

        # If any of the constraint(s) are always False (e.g. if you enter "5 < 3"), raise an error.
        # This indicates that the problem is infeasible as-written, likely because the user has frozen too
        # many decision variables using the Opti.variable(freeze=True) syntax.
        elif np.any(np.logical_not(constraint)):
            raise RuntimeError(f"""The problem is infeasible due to a constraint that always evaluates False. You 
            supplied the following constraint: {constraint}. This can happen if you've frozen too 
            many decision variables, leading to an overconstrained problem.""")

        else:  # In theory, this should never be called, so long as the constraints can be boolean-evaluated.
            raise TypeError(f"""Opti.subject_to could not determine the truthiness of your constraint, and it
            doesn't appear to be a symbolic type or a boolean type. You supplied the following constraint:
            {constraint}""")

    def parameter(self,
                  value: float = 0.,
                  ) -> cas.MX:
        """
        Initialize a new parameter.

        Args:
            value: Value to set the parameter to. Defaults to zero.
                The value can also be manually set (or re-set) after parameter initialization using the syntax:
                >>> param = opti.parameter()
                >>> opti.set_value(param, 5)
                Which initializes a new parameter and sets its value to 5.


        Returns:
            The parameter itself as a symbolic CasADi variable (MX type).

        """
        param = super().parameter()
        self.set_value(param, value)

        # Add the parameter to the parameter list to be tracked
        self.parameter_list.append(param)

        return param

    def solve(self,
              parameter_mapping: dict = None
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
        if parameter_mapping is not None:
            for k, v in parameter_mapping.items():
                self.set_value(k, v)

        return super().solve()

if __name__ == '__main__':
    pytest.main()
