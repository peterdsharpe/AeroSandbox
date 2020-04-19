import json

import casadi as cas
import numpy as np
from tqdm import tqdm

default_primal_location = 'cache/sol_primal.json'
default_dual_location = 'cache/sol_dual.json'


def save_sol_to_file(sol, save_primal=True, save_dual=True, primal_location=default_primal_location,
                     dual_location=default_dual_location):
    """
    Saves the CasADi solution to attrib_name series of JSON files.
    :param sol: A CasADi "OptiSol" object (the output of Opti.solve()).
    :param save_primal: Boolean of whether or not to save the primal solution.
    :param save_dual: Boolean of whether or not to save the dual solution.
    :param primal_location: Location of the primal JSON file.
    :param dual_location: Location of the dual JSON file.
    :return: None
    """

    # Save the primal
    if save_primal:
        sol_vals = []
        for i in tqdm(range(sol.opti.nx), desc="Saving primal variables:"):
            var = sol.opti.x[i]
            sol_vals.append(sol.value(var))
        with open(primal_location, 'w') as outfile:
            json.dump(sol_vals, outfile)

    # Save the dual
    if save_dual:
        dual_vals = []
        for i in tqdm(range(sol.opti.ng), desc="Saving dual variables:"):
            lam = sol.opti.lam_g[i]
            dual_vals.append(sol.value(lam))
        with open(dual_location, 'w') as outfile:
            json.dump(dual_vals, outfile)


def load_sol_from_file(opti, load_primal=True, load_dual=True, primal_location=default_primal_location,
                       dual_location=default_dual_location):
    """
    Loads the CasADi solution from attrib_name series of JSON files. In-place modification.
    :param opti: A CasADi "Opti" object.
    :param load_primal: Boolean of whether or not to load the primal solution.
    :param load_dual: Boolean of whether or not to load the dual solution.
    :param primal_location: Location of the primal JSON file.
    :param dual_location: Location of the dual JSON file.
    :return: None, this function modifies the opti object in-place.
    """
    # Load the primal
    if load_primal:
        with open(primal_location, 'r') as infile:
            sol_vals = json.load(infile)
        if len(sol_vals) != opti.nx:
            raise Exception(
                "Couldn't load the primal, since your problem has %i vars and the cached problem has %i vars." % (
                    opti.nx, len(sol_vals)))
        for i in tqdm(range(opti.nx), desc="Loading primal variables:"):
            opti.set_initial(opti.x[i], sol_vals[i])

    # Load the dual
    if load_dual:
        with open(dual_location, 'r') as infile:
            dual_vals = json.load(infile)
        if len(dual_vals) != opti.ng:
            raise Exception(
                "Couldn't load the dual, since your problem has %i cons and the cached problem has %i cons." % (
                    opti.ng, len(dual_vals)))
        for i in tqdm(range(opti.ng), desc="Loading dual variables:"):
            opti.set_initial(opti.lam_g[i], dual_vals[i])


def fit(
        model,  # type: callable
        x_data,  # type: dict
        y_data,  # type: np.ndarray
        param_guesses,  # type: dict
        param_bounds=None,  # type: dict
        weights=None,  # type: np.ndarray
        verbose=True,  # type: bool
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
    :return: Optimal fit parameters [dict]
    """
    opti = cas.Opti()

    # Handle weighting
    if weights is None:
        weights = cas.GenDM_ones(y_data.shape[0])
    weights /= cas.sum1(weights)

    def fit_param(initial_guess, lower_bound=None, upper_bound=None):
        """
        Helper function to create a fit variable
        :param initial_guess:
        :param lower_bound:
        :param upper_bound:
        :return:
        """
        var = opti.variable()
        opti.set_initial(var, initial_guess)
        if lower_bound is not None:
            opti.subject_to(var > lower_bound)
        if upper_bound is not None:
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

    y_model = model(x_data, params)

    residuals = y_model - y_data
    opti.minimize(cas.sum1(weights * residuals ** 2))

    opti.solver('ipopt')
    sol = opti.solve()

    params_solved = {
        k: sol.value(params[k])
        for k in params
    }

    # printing
    if verbose:
        # Print parameters
        print("\nFit Parameters:")
        if len(params_solved) <= 20:
            [print("\t%s: %f" % (k, v)) for k, v in params_solved.items()]
        else:
            print("\t%i parameters solved for." % len(params_solved))
        print("\nGoodness of Fit:")

        # Print RMS error
        weighted_RMS_error = sol.value(cas.sqrt(cas.sum1(
            weights * residuals ** 2
        )))
        print("\tWeighted RMS error: %f" % weighted_RMS_error)

        # Print R^2
        y_data_mean = cas.sum1(y_data)/y_data.shape[0]
        SS_tot = cas.sum1(weights * (y_data - y_data_mean)**2)
        SS_res = cas.sum1(weights * (y_data - y_model)**2)
        R_squared = sol.value(1 - SS_res/SS_tot)
        print("\tR^2: %f" % R_squared)

    return params_solved


sind = lambda theta: cas.sin(theta * cas.pi / 180)
cosd = lambda theta: cas.cos(theta * cas.pi / 180)
tand = lambda theta: cas.tan(theta * cas.pi / 180)
atan2d = lambda y_val, x_val: cas.atan2(y_val, x_val) * 180 / np.pi
smoothmax = lambda value1, value2, hardness: cas.log(
    cas.exp(hardness * value1) + cas.exp(hardness * value2)) / hardness  # soft maximum

# # CasADi functors (experimental)
# # dot3
# a = cas.MX.sym('a',3)
# b = cas.MX.sym('b',3)
# out = (
#         a[0] * b[0] +
#         a[1] * b[1] +
#         a[2] * b[2]
# )
# dot3 = cas.Function('dot3', [a, b], [out])
#
# # dot2
# a = cas.MX.sym('a',2)
# b = cas.MX.sym('b',2)
# out = (
#         a[0] * b[0] +
#         a[1] * b[1]
# )
# dot2 = cas.Function('dot2', [a, b], [out])
#
# # Cross
# a = cas.MX.sym('a',3)
# b = cas.MX.sym('b',3)
# out = cas.cross(a,b)
#
# del a, b, out

del default_primal_location, default_dual_location
