import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p


def beta(mach):
    return np.sqrt(1 - mach**2)


machs_to_fit = np.linspace(0.001, 0.999, 500)


def sigmoid(x):
    return 1 / (1 + np.exp(x))


def inv_sigmoid(x):
    return np.log(1 / x - 1)


weights = np.ones_like(machs_to_fit)
weights[0] = 1000
weights[machs_to_fit > 0.9] = 0.5
weights[machs_to_fit > 0.95] = 0.25


def model(x, p):
    return sigmoid(
        p["p5"] * (x - p["o5"]) ** 5
        + p["p3"] * (x - p["o3"]) ** 4
        + p["p1"] * (x - p["o1"])
    )


fit = asb.FittedModel(
    model=model,
    x_data=machs_to_fit,
    y_data=beta(machs_to_fit),
    parameter_guesses={
        "p1": 4.6,
        "o1": 0.89,
        "p3": 140,
        "o3": 0.69,
        "p5": 120,
        "o5": 0.79,
    },
    parameter_bounds={
        "o3": (0, 1),
        "o5": (0, 1),
    },
    weights=weights,
    # residual_norm_type="deviation"
)

error = fit(machs_to_fit) - beta(machs_to_fit)
error = np.array(error)

machs_to_plot = np.linspace(-0.5, 1.5, 500)
fig, ax = plt.subplots(1, 1, figsize=(8, 7), dpi=200)
plt.subplot(221)
plt.plot(machs_to_plot, beta(machs_to_plot), ".", label="")
plt.plot(machs_to_plot, fit(machs_to_plot))
plt.ylim(-0.05, 1.05)
plt.title("Fit: Normal Space")
plt.xlabel(r"Mach $M$ [-]")
plt.ylabel(r"$\beta = \sqrt{1-M^2}$")
plt.subplot(222)
plt.plot(machs_to_plot, inv_sigmoid(beta(machs_to_plot)), ".", label="")
plt.plot(machs_to_plot, inv_sigmoid(fit(machs_to_plot)))
plt.ylim(-15, 5)
plt.title("Fit: Inverse Sigmoid Space")
plt.xlabel(r"Mach $M$ [-]")
plt.ylabel(r"$\sigma^{-1}\left(\beta\right)$")
plt.subplot(223)
plt.plot(machs_to_plot, np.log(beta(machs_to_plot)), ".", label="")
plt.plot(machs_to_plot, np.log(fit(machs_to_plot)))
plt.ylim(-2.5, 0.5)
plt.title("Fit: Log Space")
plt.xlabel(r"Mach $M$ [-]")
plt.ylabel(r"$\ln(\beta)$")
plt.subplot(224)
plt.plot(machs_to_fit, error)
plt.title("Error in Fit range")
plt.savefig("machfitting.png")

p.show_plot()
