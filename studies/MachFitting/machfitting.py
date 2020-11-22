from aerosandbox.tools.fitting import fit
import matplotlib.pyplot as plt
import seaborn as sns
from aerosandbox import cas

sns.set(palette=sns.color_palette("husl"))
import numpy as np


def beta(mach):
    return np.sqrt(1 - mach ** 2)


machs_to_fit = np.linspace(0.001, 0.999, 500)


def sigmoid(x):
    return 1 / (1 + cas.exp(x))


def inv_sigmoid(x):
    return cas.log(1 / x - 1)


weights = np.ones_like(machs_to_fit)
weights[0] = 1000
weights[machs_to_fit > 0.9] = 0.5
weights[machs_to_fit > 0.95] = 0.25


# def model(x, p):
#     return sigmoid(
#         p["m"]*x+p["th"]*cas.tanh((x-p["o"]))+p["c"]
#     )
#
#
# fit_params = fit(
#     model=model,
#     x_data=machs_to_fit,
#     y_data=beta(machs_to_fit),
#     param_guesses={
#         "m": 10,
#         "c": -0.5,
#         "th": -1,
#         "o": 0.6
#     },
#     param_bounds={
#         "o": (0.25, 0.75),
#         "th": (None, 0),
#     },
#     weights=weights
# )

def model(x, p):
    return sigmoid(
        p["p5"] * (x - p["o5"]) ** 5 +
        p["p3"] * (x - p["o3"]) ** 4 +
        p["p1"] * (x - p["o1"])
    )


fit_params = fit(
    model=model,
    x_data=machs_to_fit,
    y_data=beta(machs_to_fit),
    param_guesses={
        "p1": 4.6,
        "o1": 0.89,
        "p3": 140,
        "o3": 0.69,
        "p5": 120,
        "o5": 0.79,
    },
    param_bounds={
        "o3": (0, 1),
        "o5": (0, 1),
    },
    weights=weights,
    # residual_norm_type="deviation"
)

error = model(machs_to_fit, fit_params) - beta(machs_to_fit)
error = np.array(error)

machs_to_plot = np.linspace(-0.5, 1.5, 500)
fig, ax = plt.subplots(1, 1, figsize=(8, 7), dpi=200)
plt.subplot("221")
plt.plot(machs_to_plot, beta(machs_to_plot), ".", label="")
plt.plot(machs_to_plot, model(machs_to_plot, fit_params))
plt.ylim(-0.05, 1.05)
plt.title("Fit: Normal Space")
plt.xlabel(r"Mach $M$ [-]")
plt.ylabel(r"$\beta = \sqrt{1-M^2}$")
plt.subplot("222")
plt.plot(machs_to_plot, inv_sigmoid(beta(machs_to_plot)), ".", label="")
plt.plot(machs_to_plot, inv_sigmoid(model(machs_to_plot, fit_params)))
plt.ylim(-15, 5)
plt.title("Fit: Inverse Sigmoid Space")
plt.xlabel(r"Mach $M$ [-]")
plt.ylabel(r"$\sigma^{-1}\left(\beta\right)$")
plt.subplot("223")
plt.plot(machs_to_plot, cas.log(beta(machs_to_plot)), ".", label="")
plt.plot(machs_to_plot, cas.log(model(machs_to_plot, fit_params)))
plt.ylim(-2.5, 0.5)
plt.title("Fit: Log Space")
plt.xlabel(r"Mach $M$ [-]")
plt.ylabel(r"$\ln(\beta)$")
plt.subplot("224")
plt.plot(machs_to_fit, error)
plt.title("Error in Fit range")
plt.savefig("machfitting.png")