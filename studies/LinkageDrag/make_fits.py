import pandas as pd
from read_data import df
import aerosandbox as asb
import aerosandbox.numpy as np

import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

fig, ax = plt.subplots()
Re_plot = np.geomspace(2e4, 1e6)


def model(x, p):
    Re = x["Re_l"]
    linkage_length = x["linkage_length"]
    is_covered = x["is_covered"]
    is_top = x["is_top"]

    side_drag_multiplier = np.where(
        is_top,
        p["top_drag_ratio"],
        1
    )
    covered_drag_multiplier = np.where(
        is_covered,
        p["covered_drag_ratio"],
        1
    )
    linkage_length_multiplier = 1 + p["c_length"] * linkage_length

    CDA_raw = (
            p["CD1"] / (Re / 1e5) +
            p["CD0"]
    )

    return side_drag_multiplier * covered_drag_multiplier * linkage_length_multiplier * CDA_raw


x_data = {
    col: df[col].values
    for col in df.columns if col != "CDA"
}

fit = asb.FittedModel(
    model=model,
    x_data=x_data,
    y_data=df["CDA"].values,
    parameter_guesses=dict(
        CD1=1e-3,
        CD0=1e-3,
        c_length=0,
        top_drag_ratio=1,
        covered_drag_ratio=1
    ),
    parameter_bounds=dict(
        CD0=(0, None),
        top_drag_ratio=(0, 2),
        covered_drag_ratio=(0, 2),
    ),
    verbose=False,
    put_residuals_in_logspace=True,
)
from pprint import pprint

pprint(fit.parameters)

# raise Exception

plt.loglog(
    df["Re_l"],
    df["CDA"],
    ".k",
    zorder=4,
)

n_test_data = 50000

x_data_test_ranges = dict(
    Re_l=(1e3, 1e7),
    linkage_length=(0.055, 0.085),
    is_covered=[True, False],
    is_top=[True, False]
)
x_data_test = {}
for k, v in x_data_test_ranges.items():
    if isinstance(v[0], bool):
        x_data_test[k] = np.random.rand(n_test_data) > 0.5
    elif v[0] > 0 and v[1] > 0:
        x_data_test[k] = np.exp(
            np.random.uniform(
                np.log(v[0]),
                np.log(v[1]),
                n_test_data
            )
        )
    else:
        x_data_test[k] = np.random.uniform(*v, n_test_data)

# x_data_test = {
#     k: v
#     for k, v in zip(
#         x_data_test_ranges.keys(),
#
#     )
# }
y_data_test = fit(x_data_test)
plt.plot(
    x_data_test["Re_l"],
    y_data_test,
    ".",
    alpha=0.3,
    markersize=1,
)

# plt.loglog(
#     Re_plot,
#     fit(dict(
#         Re=Re_plot,
#         is_top=True,
#     )),
#     "-",
#     alpha=0.6,
#     color=line.get_color()
# )

# print(f"{name.rjust(30)} : lambda Re: {fit.parameters['a']:.8f} * (Re / 1e5) ** {fit.parameters['b']:.8f}")

# plt.xlim(left=1e4, right=2e6)
p.show_plot(
    "Linkage Drag",
    "Reynolds Number [-]",
    "Drag Area [m$^2$]"
)
