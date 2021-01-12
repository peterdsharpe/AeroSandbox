import numpy as np
import casadi as cas
from aerosandbox.tools.string_formatting import stdout_redirected
from aerosandbox.optimization.opti import Opti
from typing import Union, Dict, Callable


def fit(
        model: Callable[
            [
                Union[Dict[str, np.ndarray], np.ndarray],
                Dict[str, float]
            ],
            np.ndarray
        ],
        x_data: Union[Dict[str, np.ndarray], np.ndarray],
        y_data: np.ndarray,
        param_guesses: Dict[str, float],
        param_bounds: Dict[str, tuple] = None,
        weights: np.ndarray = None,
        verbose: bool = True,
        scale_problem: bool = True,
        put_residuals_in_logspace: bool = False,
        residual_norm_type: str = "SSE",
        plot_fit: bool = False,
):
    """
    Fits an analytical model to n datapoints using an automatic-differentiable optimization approach.

    For examples of usage, see aerosandbox.tools.test_fitting

    :param model: A callable with syntax f(x, p) where:
            * x is a dict of dependent variables.
              If one-dimensional (e.g. f(x) instead of f(x,y)), you can instead supply x as a 1D ndarray.
                Same format as x_data [dict of 1D ndarrays of length n].
            * p is a dict of parameters. Same format as param_guesses [dict with syntax param_name:param_value].
        Model should return a 1D ndarray of length n.
        Model should use CasADi functions for differentiability.
    :param x_data: a dict of dependent variables.
        * If one-dimensional (e.g. f(x) instead of f(x,y)), you can instead supply x_data as a 1D ndarray.
        * Same format as model's x. [1D ndarray or dict of 1D ndarrays of length n]
    :param y_data: independent variable. [1D ndarray of length n]
    :param param_guesses: a dict of fit parameters.
        * Same format as model's p [dict with syntax param_name:param_value].
        * Keys are parameter names, values are initial guesses.
    :param param_bounds: Optional: a dict of bounds on fit parameters.
        Keys are parameter names, values are a tuple of (min, max).
        May contain only a subset of param_guesses if desired.
        Use None to represent one-sided constraints (i.e. (None, 5)).
        [dict of parameter_name:tuples]
    :param weights: Optional: weights for data points. If not supplied, weights are assumed to be uniform.
        Weights are automatically normalized. [1D ndarray of length n]
    :param verbose: Whether or not to print information about parameters and goodness of fit.
    :param scale_problem: Whether or not to attempt to scale variables, constraints, and objective for more robust solve. [boolean]
    :param put_residuals_in_logspace: Whether to optimize using the logarithmic error as opposed to the absolute error
        (useful for minimizing percent error).
        Note: If any model outputs or data are negative, this will fail!
    :param residual_norm_type: What type of error norm should we use to optimize the fit parameters? [string]
        Options:
            * "SSE": minimizes the sum of squared errors.
            * "deviation": minimizes the maximum deviation of the fit; basically the infinity-norm of the error vector.
    :return: Optimal fit parameters [dict with same keys as param_guesses]
    """
    ### Initialize an optimization environment
    opti = Opti()

    ### Flatten all inputs
    def flatten(input):
        return np.array(input).flatten()

    try:
        x_data = {
            k: flatten(v)
            for k, v in x_data.items()
        }
        x_data_is_dict = True
    except AttributeError:  # If it's not a dict or dict-like, assume it's a 1D ndarray dataset
        x_data = flatten(x_data)
        x_data_is_dict = False
    y_data = flatten(y_data)
    n_datapoints = len(y_data)

    ### Handle weighting
    if weights is None:
        weights = np.ones(n_datapoints)
    else:
        weights = flatten(weights)
    weights /= np.sum(weights)

    ### Check format of param_bounds input
    if param_bounds is None:
        param_bounds = {}
    for k, v in param_bounds.items():
        if k not in param_guesses.keys():
            raise ValueError(f"A parameter name (key = \"{k}\") in param_bounds was not found in param_guesses.")
        if not len(v) == 2:
            raise ValueError("Every value in param_bounds must be a tuple in the format (lower_bound, upper_bound). "
                             "For one-sided bounds, use None for the unbounded side.")

    ### Check dimensionality of inputs to fitting algorithm
    relevant_inputs = {
        "y_data" : y_data,
        "weights": weights,
    }
    try:
        relevant_inputs.update(x_data)
    except TypeError:
        relevant_inputs.update({"x": x_data})

    for key, value in relevant_inputs.items():
        # Check that the length of the inputs are consistent
        series_length = len(value)
        if not series_length == n_datapoints:
            raise ValueError(
                f"The supplied data series \"{key}\" has length {series_length}, but y_data has length {n_datapoints}.")

    ### Set up the variables and bounds constraints

    def fit_param(initial_guess, lower_bound=None, upper_bound=None):
        """
        Helper function to create a fit variable
        :param initial_guess:
        :param lower_bound:
        :param upper_bound:
        :return:
        """
        if scale_problem and np.abs(initial_guess) > 1e-8:
            var = opti.variable(scale=np.abs(initial_guess), init_guess=initial_guess)  # scale variables
        else:
            var = opti.variable(init_guess=initial_guess)
        if lower_bound is not None:
            lower_bound_abs = np.abs(lower_bound)
            if scale_problem and lower_bound_abs > 1e-8:
                opti.subject_to(var / lower_bound_abs > lower_bound / lower_bound_abs)
            else:
                opti.subject_to(var > lower_bound)
        if upper_bound is not None:
            upper_bound_abs = np.abs(upper_bound)
            if scale_problem and upper_bound_abs > 1e-8:
                opti.subject_to(var / upper_bound_abs < upper_bound / upper_bound_abs)
            else:
                opti.subject_to(var < upper_bound)
        return var

    params = {
        k: fit_param(param_guesses[k]) if k not in param_bounds else
        fit_param(param_guesses[k], param_bounds[k][0], param_bounds[k][1])
        for k in param_guesses
    }

    ### Setup the objective handling
    if residual_norm_type == "SSE":
        def SSE_objective(params):
            """
            Given some parameters for the model, what is the "badness" of the corresponding fit.

            Args:
                params

            Returns: A scalar representing the "badness" of the fit.

            """
            y_model = model(x_data, params)
            if y_model is None:
                raise TypeError("model(x, param_guesses) returned None, when it should've returned a 1D ndarray.")

            if put_residuals_in_logspace:
                residuals = cas.log(y_model) - cas.log(y_data)
            else:
                residuals = y_model - y_data

            return cas.sum1(weights * residuals ** 2)

        initial_objective = SSE_objective(param_guesses)
        objective = SSE_objective(params)

    elif residual_norm_type == "deviation":
        y_model_initial = model(x_data, param_guesses)
        initial_objective = np.max(np.abs(y_model_initial - y_data))

        y_model = model(x_data, params)

        if scale_problem and initial_objective >= 1e-8:
            objective = opti.variable(init_guess=initial_objective, scale=initial_objective)
            opti.subject_to([
                objective / initial_objective >= (y_model - y_data) / initial_objective,
                objective / initial_objective >= -(y_model - y_data) / initial_objective
            ])
        else:
            objective = opti.variable(init_guess=initial_objective)
            opti.subject_to([
                objective >= (y_model - y_data),
                objective >= -(y_model - y_data)
            ])


    else:
        return ValueError("Bad input for the 'residual_type' parameter.")

    if scale_problem:
        opti.minimize(objective / initial_objective)
    else:
        opti.minimize(objective)

    # Solve
    if verbose:
        sol = opti.solve()
    else:
        with stdout_redirected():
            sol = opti.solve()

    params_solved = {}
    for k in params:
        try:
            params_solved[k] = sol.value(params[k])
        except:
            params_solved[k] = np.NaN

    # printing
    if verbose:
        # Print parameters
        print("\nFit Parameters:")
        if len(params_solved) <= 20:
            [print("\t%s = %f" % (k, v)) for k, v in params_solved.items()]
        else:
            print("\t%i parameters solved for." % len(params_solved))
        print("\nGoodness of Fit:")

        # Print objective function value
        print(f"\tInitial Badness (objective function): {initial_objective}")
        print(f"\tFinal Badness (objective function): {sol.value(objective)}")

        # Print R^2
        y_data_mean = cas.sum1(y_data) / y_data.shape[0]
        SS_tot = cas.sum1(weights * (y_data - y_data_mean) ** 2)
        SS_res = cas.sum1(weights * (y_data - model(x_data, params_solved)) ** 2)
        R_squared = sol.value(1 - SS_res / SS_tot)
        print(f"\tR^2: {R_squared}")

    return params_solved
