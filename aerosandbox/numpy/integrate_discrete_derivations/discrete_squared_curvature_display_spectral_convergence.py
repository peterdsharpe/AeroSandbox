import aerosandbox as asb
import aerosandbox.numpy as np
import sympy as s
from aerosandbox.numpy.integrate_discrete import integrate_discrete_squared_curvature

n_samples = 10000

x = s.symbols("x")
k = s.symbols("k", integer=True, positive=True)
f = s.cos(k * x * 2 * s.pi) / k ** 2
d2fdx2 = f.diff(x, 2)
exact = s.simplify(s.integrate(d2fdx2 ** 2, (x, 0, 1)))

print(float(exact))

def get_approx(period=10, method="cubic"):
    x_vals = np.linspace(0, 1, n_samples).astype(float)
    f_vals = s.lambdify(
        x,
        f.subs(k, (n_samples - 1) / period),
    )(x_vals)

    # import matplotlib.pyplot as plt
    # import aerosandbox.tools.pretty_plots as p
    # fig, ax = plt.subplots()
    # ax.plot(x_vals, f_vals, ".-")
    # p.show_plot()

    approx = np.sum(integrate_discrete_squared_curvature(
        f=f_vals,
        x=x_vals,
        method=method,
    ))

    return approx

periods = np.geomspace(1, 10000, 201)
# approxes = get_approx(periods)
# rel_errors = np.abs(approxes - float(exact)) / float(exact)

import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
fig, ax = plt.subplots(2, 1, figsize=(6, 8))
for method in ["cubic", "simpson", "hybrid_simpson_cubic"]:
    approxes = np.vectorize(
        lambda period: get_approx(period, method=method)
    )(periods)
    rel_errors = np.abs(approxes - float(exact)) / float(exact)
    ax[0].loglog(periods, rel_errors, ".-", label=method, alpha=0.8)
    ax[1].semilogx(periods, approxes, ".-", label=method, alpha=0.8)
plt.xlabel("Period [samples]")
ax[0].set_ylabel("Relative Error")
ax[1].set_ylabel("Approximation")
plt.sca(ax[1])
p.hline(
    float(exact),
    color="k",
    linestyle="--",
    label="Exact",
    alpha=0.5
)

p.show_plot()