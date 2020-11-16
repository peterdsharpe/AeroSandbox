import numpy as np
import casadi as cas
from aerosandbox.tools.string_formatting import stdout_redirected


def fit(
        model,  # type: callable
        x_data,  # type: dict
        y_data,  # type: np.ndarray
        param_guesses,  # type: dict
        param_bounds=None,  # type: dict
        weights=None,  # type: np.ndarray
        verbose=True,  # type: bool
        scale_problem=True,  # type: bool
        put_residuals_in_logspace=False,  # type: bool
):
    """
    Fits a model to data through least-squares minimization.
    :param model: A callable with syntax f(x, p) where:
            x is a dict of dependent variables. Same format as x_data [dict of 1D ndarrays of length n].
            p is a dict of parameters. Same format as param_guesses [dict of scalars].
        Model should use CasADi functions for differentiability.
    :param x_data: a dict of dependent variables. Same format as model's x. [dict of 1D ndarrays of length n]
    :param y_data: independent variable. [1D ndarray of length n]
    :param param_guesses: a dict of fit parameters. Same format as model's p. Keys are parameter names, values are initial guesses. [dict of scalars]
    :param param_bounds: Optional: a dict of bounds on fit parameters.
        Keys are parameter names, values are a tuple of (min, max).
        May contain only a subset of param_guesses if desired.
        Use None to represent one-sided constraints (i.e. (None, 5)).
        [dict of tuples]
    :param weights: Optional: weights for data points. If not supplied, weights are assumed to be uniform.
        Weights are automatically normalized. [1D ndarray of length n]
    :param verbose: Whether or not to print information about parameters and goodness of fit.
    :param scale_problem: Whether or not to attempt to scale variables, constraints, and objective for more robust solve. [boolean]
    :param put_residuals_in_logspace: Whether to optimize using the logarithmic error as opposed to the absolute error (useful for minimizing percent error).
        Note: If any model outputs or data are negative, this will fail!
    :return: Optimal fit parameters [dict]
    """
    opti = cas.Opti()

    # Handle weighting
    if weights is None:
        weights = cas.GenDM_ones(y_data.shape[0])
    weights /= cas.sum1(weights)

    # Check dimensionality of inputs to fitting algorithm
    n_datapoints = len(np.array(y_data).flatten())
    for key, value in {
        **x_data,
        "y_data": y_data,
        "weights": weights,
    }.items():
        series = np.array(value)

        # Check that it's 1D or 1D-ish
        shape = np.shape(series)
        if len(shape) == 1:
            pass
        else:
            number_of_nontrivial_dimensions = sum([
                dimension_length != 1 for dimension_length in shape
            ])
            if number_of_nontrivial_dimensions != 1:
                raise ValueError(f"The supplied data series \"{key}\" is not a 1D ndarray (or convertible to one). You should flatten it.")
            series = series.flatten()

        # Check that the length of the inputs are consistent
        series_length = len(series)
        if not series_length == n_datapoints:
            raise ValueError(f"The supplied data series \"{key}\" has length {series_length}, but y_data has length {n_datapoints}.")




    def fit_param(initial_guess, lower_bound=None, upper_bound=None):
        """
        Helper function to create a fit variable
        :param initial_guess:
        :param lower_bound:
        :param upper_bound:
        :return:
        """
        if scale_problem and np.abs(initial_guess) > 1e-8:
            var = initial_guess * opti.variable()  # scale variables
        else:
            var = opti.variable()
        opti.set_initial(var, initial_guess)
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

    if param_bounds is None:
        params = {
            k: fit_param(param_guesses[k])
            for k in param_guesses
        }
    else:
        params = {
            k: fit_param(param_guesses[k]) if k not in param_bounds else
            fit_param(param_guesses[k], param_bounds[k][0], param_bounds[k][1])
            for k in param_guesses
        }

    if scale_problem:
        y_model_initial = model(x_data, param_guesses)
        if not put_residuals_in_logspace:
            residuals_initial = y_model_initial - y_data
        else:
            residuals_initial = cas.log10(y_model_initial) - cas.log10(y_data)
        SSE_initial = cas.sum1(weights * residuals_initial ** 2)

        y_model = model(x_data, params)
        if not put_residuals_in_logspace:
            residuals = y_model - y_data
        else:
            residuals = cas.log10(y_model) - cas.log10(y_data)
        SSE = cas.sum1(weights * residuals ** 2)
        opti.minimize(SSE / SSE_initial)

    else:
        y_model = model(x_data, params)
        if not put_residuals_in_logspace:
            residuals = y_model - y_data
        else:
            residuals = cas.log10(y_model) - cas.log10(y_data)
        SSE = cas.sum1(weights * residuals ** 2)
        opti.minimize(SSE)

    # Solve
    p_opts = {}
    s_opts = {}
    s_opts["max_iter"] = 3e3  # If you need to interrupt, just use ctrl+c
    # s_opts["mu_strategy"] = "adaptive"
    opti.solver('ipopt', p_opts, s_opts)
    opti.solver('ipopt')
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

        # Print RMS error
        weighted_RMS_error = sol.value(cas.sqrt(cas.sum1(
            weights * residuals ** 2
        )))
        print("\tWeighted RMS error: %f" % weighted_RMS_error)

        # Print R^2
        y_data_mean = cas.sum1(y_data) / y_data.shape[0]
        SS_tot = cas.sum1(weights * (y_data - y_data_mean) ** 2)
        SS_res = cas.sum1(weights * (y_data - y_model) ** 2)
        R_squared = sol.value(1 - SS_res / SS_tot)
        print("\tR^2: %f" % R_squared)

    return params_solved
