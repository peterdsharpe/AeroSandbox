import aerosandbox as asb
import aerosandbox.numpy as np
import sympy as s
from aerosandbox.numpy.integrate_discrete import integrate_discrete_squared_curvature

n_samples = 10000

x = s.symbols("x", real=True)
k = s.symbols("k", positive=True, real=True)
f = s.cos(k * x * 2 * s.pi) / k ** 2
d2fdx2 = f.diff(x, 2)
exact = s.simplify(s.integrate(d2fdx2 ** 2, (x, 0, 1)))


@np.vectorize
def get_approx(period=10, method="cubic"):
    x_vals = np.linspace(0, 1, n_samples).astype(float)
    f_vals = s.lambdify(
        x,
        f.subs(k, (n_samples - 1) / period),
    )(x_vals)

    approx = np.sum(integrate_discrete_squared_curvature(
        f=f_vals,
        x=x_vals,
        method=method,
    ))

    return approx


periods = np.geomspace(2, n_samples, 1001)
exacts = s.lambdify(k, exact)((n_samples - 1) / periods)

import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

fig, ax = plt.subplots(2, 1, figsize=(6, 8))
for method in ["cubic", "simpson", "hybrid_simpson_cubic"]:
    approxes = get_approx(periods, method)
    ratio = approxes / exacts
    rel_errors = np.abs(approxes - exacts) / exacts
    ax[0].loglog(periods, rel_errors, ".-", label=method, alpha=0.8, markersize=3)
    ax[1].semilogx(periods, ratio, ".-", label=method, alpha=0.8, markersize=3)
plt.xlabel("Period [samples]")
ax[0].set_ylabel("Relative Error")
ax[1].set_ylabel("Approximation / Exact")
plt.sca(ax[1])
ax[1].set_ylim(bottom=0)
ax[1].plot(
    periods,
    np.ones_like(periods),
    label="Exact",
    color="k",
    linestyle="--",
    alpha=0.5
)
p.show_plot()
