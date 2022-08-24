from read_data import df
import aerosandbox as asb
import aerosandbox.numpy as np

import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

fig, ax = plt.subplots()
Re_plot = np.geomspace(2e4, 1e6)

for name, series in df.iterrows():
    fit = asb.FittedModel(
        model=lambda x, p: p['a'] * (x / 1e5) ** p['b'],
        x_data=series.index.values,
        y_data=series.values,
        parameter_guesses=dict(
            a=1,
            b=1
        ),
        verbose=False
    )
    line, = plt.loglog(
        series.index.values,
        series.values,
        ".",
        label=name
    )
    plt.loglog(
        Re_plot,
        fit(Re_plot),
        "-",
        color=line.get_color()
    )

    print(f"{name.rjust(30)} : lambda Re: {fit.parameters['a']:.8f} * (Re / 1e5) ** {fit.parameters['b']:.8f}")


plt.xlim(left=1e4, right=2e6)
p.show_plot(
    "Linkage Drag",
    "Reynolds Number [-]",
    "Drag Area [m$^2$]"
)
