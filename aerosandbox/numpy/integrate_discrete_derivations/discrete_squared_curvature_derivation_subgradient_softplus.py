import aerosandbox.numpy as np
from scipy import interpolate, integrate
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
import sympy as s


def func_num(x, softness=1):
    return np.softmax(
        0,
        x,
        softness=softness
    )


def func_sym(x, softness=1):
    return s.log(
        1 + s.exp(
            x / softness
        )
    ) * softness
    # return s.sqrt(x ** 2 + softness ** 2)



softness = s.Rational(1, 100)

x = np.concatenate([
    np.sinspace(0, 200 * softness, 1000)[::-1][:-1] * -1,
    np.sinspace(0, 200 * softness, 1000),
]).astype(float)

# Discrete chirp function
f = func_num(x, softness=float(softness))
#
# f_interp = interpolate.InterpolatedUnivariateSpline(x, f, k=3)
# exact = integrate.quad(
#     lambda x: f_interp.derivative(2)(x) ** 2,
#     x[0], x[-1],
#     epsrel=1e-6
# )[0]

x_sym = s.symbols("x")
f_sym = func_sym(x_sym, softness=softness)
exact = s.integrate(
    f_sym.diff(x_sym, 2) ** 2,
    (x_sym, -s.oo, s.oo)
)

print(f"Exact: {exact}")

# Estimate discrete
slopes = np.diff(f) / np.diff(x)

subgradients = np.diff(slopes)

discrete = np.sum(subgradients ** 2)

print(f"Discrete: {discrete}")

print(f"Ratio: {discrete / exact} (equiv: 1 / {exact / discrete})")
