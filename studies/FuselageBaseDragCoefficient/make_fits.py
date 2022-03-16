from aerosandbox.tools.webplotdigitizer_reader import read_webplotdigitizer_csv
import aerosandbox as asb
import aerosandbox.numpy as np

data = read_webplotdigitizer_csv("mach_vs_base_drag_coefficient.csv")["data"]

import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

fig, ax = plt.subplots()
plt.plot(data[:, 0], data[:, 1], ".")


def model(m, p):
    return np.blend(
        p["trans_str"] * (m - p["m_trans"]),
        p["pc_sup"] + p["a"] * np.exp(-(p["scale_sup"] * (m - p["center_sup"])) ** 2),
        p["pc_sub"]
    )


fit = asb.FittedModel(
    model=model,
    x_data=data[:, 0],
    y_data=data[:, 1],
    parameter_guesses={
        "a"        : 0.2,
        "scale_sup"    : 1,
        "center_sup"    : 1,
        "m_trans"  : 1.,
        "trans_str": 5,
        "pc_sub"   : 0.16,
        "pc_sup"   : 0.05,
    },
    parameter_bounds={
        "trans_str": (0, 10),
        # "m_trans": (1,1),
        # "center_sup": (0, None)
    },
    # weights = data[:, 0] > 1,
    verbose=False
)

print(fit.parameters)

mach = np.linspace(0, 5, 1000)
plt.plot(mach, fit(mach), "-k", label="Fit")
plt.ylim(0, 0.2)

p.show_plot(
    "Fuselage Base Drag Coefficient",
    "Mach [-]",
    "Fuselage Base Drag Coefficient [-]"
)
