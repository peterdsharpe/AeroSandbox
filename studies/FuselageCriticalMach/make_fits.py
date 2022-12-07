from aerosandbox.tools.webplotdigitizer_reader import read_webplotdigitizer_csv
import aerosandbox as asb
import aerosandbox.numpy as np

data = read_webplotdigitizer_csv(filename="data.csv")
sub = data["Subsonic"]
sup = data["Supersonic"]

import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

fig, ax = plt.subplots()
plt.plot(sub[:, 0], sub[:, 1], ".", label="Subsonic Designs")
plt.plot(sup[:, 0], sup[:, 1], ".", label="Supersonic Designs")

fr = np.linspace(0., 15, 500)


def model(fr, p):
    """
    Using this model because it satisfies some things that should be true in asymptotic limits:

    As the fineness ratio goes to infinity, the drag-divergent Mach should go to 1.

    As the fineness ratio goes to 0, the drag-divergent Mach should go to some reasonable value in the range of 0 to
    1, probably around 0.5? Certainly no more than 0.6, I imagine. (intuition)

    """
    return 1 - (p["a"] / (fr + p["b"])) ** p["c"]


fit = asb.FittedModel(
    model=model,
    x_data=np.concatenate([sub[:, 0], sup[:, 0]]),
    y_data=np.concatenate([sub[:, 1], sup[:, 1]]),
    weights=np.concatenate([
        np.ones(len(sub)) / len(sub),
        np.ones(len(sup)) / len(sup),
    ]),
    parameter_guesses={
        "a": 0.5,
        "b": 3,
        "c": 1,
    },
    parameter_bounds={
        "a": (0, None),
        "b": (0, None),
        "c": (0, None)
    },
    residual_norm_type="L2"
)

plt.plot(fr, fit(fr), "-k", label="Fit")

p.show_plot(
    "Drag-Divergent Mach Number for Generic\nFuselage of Varying Fineness Ratio",
    "$2L_n / d$",
    r"$\mathrm{Mach}_{DD}$ [-]"
)
print(fit.parameters)
