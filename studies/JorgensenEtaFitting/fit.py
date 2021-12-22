import aerosandbox as asb
import aerosandbox.numpy as np
import pandas as pd

data = pd.read_csv("fineness-vs-eta.csv", names=["fineness", "eta"])
fineness = data['fineness'].values
eta = data['eta'].values


def model(x, p):
    return 1 - p["1scl"] / (x - p["1cen"]) - (p["2scl"] / (x - p["2cen"])) ** 2


fit = asb.FittedModel(
    model=model,
    x_data=fineness,
    y_data=eta,
    parameter_guesses={
        "1scl": 1,
        "1cen": -25,
        "2scl": -1,
        "2cen": -25,
    },
    parameter_bounds={
        "1scl": (0, None),
        "1cen": (None, 0),
        "2scl": (0, None),
        "2cen": (None, 0),
    },
    residual_norm_type="L2",
    verbose=False
)
print(fit.parameters)

from aerosandbox.tools.pretty_plots import plt, show_plot

fig, ax = plt.subplots()
plt.plot(fineness, eta, ".k")
fineness_plot = np.linspace(0, 40, 500)
plt.plot(fineness_plot, fit(fineness_plot), "-")
show_plot(r"$\eta$ Fitting", "Fineness Ratio", r"$\eta$")
