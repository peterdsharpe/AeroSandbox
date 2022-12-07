from aerosandbox.tools.webplotdigitizer_reader import read_webplotdigitizer_csv
import aerosandbox as asb
import aerosandbox.numpy as np

data = read_webplotdigitizer_csv("mach_vs_Cd.csv")["data"]

import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

fig, ax = plt.subplots()
plt.plot(data[:, 0], data[:, 1], ".")


def model(m, p):
    return np.blend(
        p["trans_str"] * (m - p["trans"]),
        p["cd_sup"] + np.exp(p["a_sup"] + p["s_sup"] * (m - p["trans"])),
        p["cd_sub"] + np.exp(p["a_sub"] + p["s_sub"] * (m - p["trans"]))
    )


fit = asb.FittedModel(
    model=model,
    x_data=data[:, 0],
    y_data=data[:, 1],
    parameter_guesses={
        "trans"    : 1,
        "trans_str": 10,
        "cd_sub"   : 1.2,
        "cd_sup"   : 1.2,
        "a_sub"    : -0.4,
        "a_sup"    : -0.4,
        "s_sub"    : 4.6,
        "s_sup"    : -1.5,
    },
    parameter_bounds={
        "s_sub"    : (0, None),
        "s_sup"    : (None, 0),
        # "cd_sub": (1.2, 1.2),
        "trans"    : (0.9, 1.1),
        "trans_str": (0, None),
    },
    weights=1 + 5 * ((data[:, 0] > 0.9) & (data[:, 0] < 1.1)),
    verbose=False,
    # residual_norm_type="LInf",
    put_residuals_in_logspace=True
)

from pprint import pprint

pprint(fit.parameters)

mach = np.linspace(0, 5, 1000)
plt.plot(mach, fit(mach), "-k", label="Fit")
plt.ylim(1, 2.5)

p.show_plot(
    "Mach vs. Cylinder Drag",
    "Mach [-]",
    "Cylinder Drag Coefficient $C_d$"
)
