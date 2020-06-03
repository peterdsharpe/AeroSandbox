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


sind = lambda theta: cas.sin(theta * cas.pi / 180)
cosd = lambda theta: cas.cos(theta * cas.pi / 180)
tand = lambda theta: cas.tan(theta * cas.pi / 180)
atan2d = lambda y_val, x_val: cas.atan2(y_val, x_val) * 180 / np.pi
clip = lambda x, min, max: cas.fmin(cas.fmax(min, x), max)

def smoothmax(value1, value2, hardness):
    """
    A smooth maximum between two functions.
    Useful because it's differentiable and convex!
    Great writeup by John D Cook here:
        https://www.johndcook.com/soft_maximum.pdf
    :param value1: Value of function 1.
    :param value2: Value of function 2.
    :param hardness: Hardness parameter. Higher values make this closer to max(x1, x2).
    :return: Soft maximum of the two supplied values.
    """
    value1 = value1 * hardness
    value2 = value2 * hardness
    max = cas.fmax(value1, value2)
    min = cas.fmin(value1, value2)
    out = max + cas.log(1 + cas.exp(min - max))
    out /= hardness
    return out


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

if __name__ == '__main__':
    # Test smoothmax
    import matplotlib.pyplot as plt
    from matplotlib import style
    import seaborn as sns

    sns.set(font_scale=1)
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
    x = np.linspace(-10, 10, 100)
    y1 = x
    y2 = -2 * x - 3
    hardness = 0.5
    plt.plot(x, y1, label="y1")
    plt.plot(x, y2, label="y2")
    plt.plot(x, smoothmax(y1, y2, hardness), label="smoothmax")
    plt.xlabel(r"x")
    plt.ylabel(r"y")
    plt.title(r"Smoothmax")
    plt.tight_layout()
    plt.legend()
    # plt.savefig("C:/Users/User/Downloads/temp.svg")
    plt.show()
