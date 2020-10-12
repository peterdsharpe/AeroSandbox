import casadi as cas
from typing import Union, List
import numpy as np
import pytest


class Opti(cas.Opti):
    def __init__(self,
                 variable_categories_to_freeze=[],
                 ):
        super().__init__()
        self.solver('ipopt')  # Default to IPOPT solver
        self.categories_to_freeze = variable_categories_to_freeze

    def variable(self,
                 n_vars: int = 1,
                 init_guess: Union[float, np.ndarray] = None,
                 scale: float = 1,
                 log_transform: bool = False,
                 category: str = "Default",
                 fix: bool = False,
                 ):
        """
        Initializes a new decision variable.

        Args:
            n_vars: Number of variables to initialize (used to initialize a vector of variables). If you are
                initializing a scalar variable (the most typical case), leave this equal to 1. When using vector variables,
                individual components of this vector of variables can be accessed via indexing.

                Example:
                    >>> opti = asb.Opti()
                    >>> my_var = opti.variable(n_vars = 5)
                    >>> opti.subject_to(my_var[3] >= my_var[2])  # This is a valid way of indexing
                    >>> my_sum = asb.cas.sum1(my_var)  # This will sum up all elements of `my_var`

            init_guess: Initial guess for the variable being initialized. For scalar variables, this should be a
                float. For vector variables (see `n_vars`), you can provide either a float (in which case all elements
                of the vector will be initialized to the given value) or an iterable of equal length (in which case
                each element will be initialized to the corresponding value in the given iterable).

                In the case where the variable is to be log-transformed (see `log_transform`), the initial guess should
                not be log-transformed as well; this happens under the hood. The initial guess must, of course, be a
                positive number in this case.

                If not specified, initial guess defaults to 0 for non-log-transformed variables and 1 for
                log-transformed variables.

            scale:

            log_transform:
            category:

        Returns:

        """
        # Set defaults
        if init_guess is None:
            if log_transform:
                init_guess = 1
            else:
                init_guess = 0

        # Validate the inputs
        if np.any(scale <= 0):
            raise ValueError("The 'scale' argument must be a positive number.")

        # If the variable is in a category to be frozen, fix the variable at the initial guess.
        if category in self.categories_to_freeze:
            fix = True

        # If the variable is to be fixed, just return the initial guess and end here
        if fix:
            return init_guess * np.ones(n_vars)

        if not log_transform:
            var = scale * super().variable(n_vars)
            self.set_initial(var, init_guess)
        else:
            if np.any(init_guess <= 0):
                raise ValueError(
                    "If you are initializing a log-transformed variable, the initial guess must be positive.")
            log_scale = scale / init_guess
            log_var = log_scale * super().variable(n_vars)
            var = cas.exp(log_var)
            self.set_initial(log_var, cas.log(init_guess))

        return var

    def subject_to(self,
                   constraints: List,
                   ):

        # Put the constraints into a list, if they aren't already in one.
        if not isinstance(constraints, List):
            constraints = [constraints]

        # Iterate through the constraints
        for constraint in constraints:
            # If it's a proper constraint, pass it into the problem formulation and be done with it.
            if isinstance(constraint, cas.MX):
                super().subject_to(constraint)
                continue

            # If the constraint(s) always evaluates True (e.g. if you enter "5 > 3"), skip it.
            # This allows you to toggle fixed variables without causing problems with setting up constraints.
            if np.all(constraint):
                continue

            # If any of the constraint(s) are always False (e.g. if you enter "5 < 3"), raise an error.
            # This indicates that the problem is infeasible as-written, likely because the user has fixed too
            # many decision variables using the Opti.variable(fix=True) syntax.
            elif np.any(np.logical_not(constraint)):
                raise RuntimeError("""The problem is infeasible due to a constraint that always evaluates False.
                                   Check if you've fixed too many decision variables, leading to an overconstrained 
                                   problem.""")

            else:  # In theory, this should never be called, so long as the constraints can be boolean-evaluated.
                raise TypeError(f"""Opti.subject_to could not determine the truthiness of your constraint, and it
                doesn't appear to be a symbolic type. You supplied the following constraint:
                {constraint}""")


if __name__ == '__main__':
    pytest.main()
