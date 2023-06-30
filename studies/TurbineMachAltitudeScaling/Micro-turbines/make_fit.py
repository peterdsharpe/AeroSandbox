import aerosandbox as asb
import aerosandbox.numpy as np
from read_data import df
import aerosandbox.tools.units as u


def model(x, p):
    sdr = x["stagnation_density_ratio"]
    mach_squared = x["mach"] ** 2
    return (
            (p["a"] - p["b"] * mach_squared * (1 + p["machslope"] * sdr)) *
            sdr ** (p["sdr"])
            # sdr
    )


fit = asb.FittedModel(
    model=model,
    x_data={
        "mach"                    : df["mach"],
        "stagnation_density_ratio": df["stagnation_density_ratio"]
    },
    y_data=df["thrust_over_nominal_thrust"],
    parameter_guesses={
        "a"  : 1,
        "sdr": 1,
        "b"  : 0,
        "machslope" : 0,
    },
    residual_norm_type="L1",
    # put_residuals_in_logspace=True,
    verbose=False,
)

print(fit.parameters)
scalefactor = fit.parameters["a"]

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    cm = plt.get_cmap("crest")
    # cm = p.sns.husl_palette(as_cmap=True)
    norm = plt.Normalize(0, 1)

    fig, ax = plt.subplots()
    plt.xscale('log')
    plt.yscale('log')

    plt.scatter(
        df["stagnation_density_ratio"],
        df["thrust_over_nominal_thrust"] / scalefactor,
        s=10,
        c=df["mach"],
        cmap=cm,
        norm=norm,
        edgecolors="k",
        linewidths=0.25,
        alpha=0.6
    )

    for mach in [0, 0.3, 0.6, 0.9]:
        altitudes = np.linspace(0, 40000 * u.foot)
        atmo = asb.Atmosphere(altitude=altitudes)
        op_point = asb.OperatingPoint(
            atmosphere=atmo,
            velocity=mach * atmo.speed_of_sound()
        )
        sea_level_atmo = asb.Atmosphere()
        stagnation_density_ratios = (
                (op_point.total_pressure() / atmo.temperature()) /
                (sea_level_atmo.pressure() / sea_level_atmo.temperature())
        )

        plt.plot(
            stagnation_density_ratios,
            fit({
                "mach"                    : mach,
                "stagnation_density_ratio": stagnation_density_ratios,
            }) / scalefactor,
            label=f"M{mach:.1f}",
            color=cm(norm(mach)),
            alpha=0.6,
        )

    p.labelLines(
        lines=ax.lines,
        fontsize=8,
        alpha=0.8
    )
    for line in ax.lines:
        line.set_label('_')


    plt.plot([], [], "-k", label="New Model, at various Machs")
    plt.plot([], [], ".k", markersize=6, label="Flight Test Data")

    plt.annotate(
        text="Data from Hamilton Sundstrand, AIAA 2003-6568",
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        ha="left",
        fontsize=9,
        alpha=0.5,
    )

    plt.annotate(
        text="$\\longleftarrow$ varying altitude $\\longrightarrow$\n\n",
        xy=(0.6, 0.6),
        rotation=36.5,
        ha="center",
        va="center",
        alpha=0.7,
        fontsize=10,
    )
    

    plt.plot([1e-3, 2], [1e-3, 2], ":k", label="Previous Model", alpha=0.6)

    import matplotlib as mpl

    ax.tick_params(which="major", labelsize=10)

    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    plt.colorbar(
        label="Mach Number [-]",
        mappable=plt.cm.ScalarMappable(
            norm=norm,
            cmap=cm
        )
    )

    plt.xlim(0.3, 1.8)
    plt.ylim(0.25, 1.5)

    p.show_plot(
        "Effect of Transonic Mach and High Altitude\non Micro-Turbojet Nondimensional Performance",
        "Freestream Stagnation Density Ratio\n($\\rho_{t0}$) / (Sea-level Static $\\rho$)",
        "Thrust ratio\n(Thrust) / (Sea-level Static Thrust)",
        # legend=False
        dpi=300
    )

    ####################################################################

    fig, ax = plt.subplots()

    Machs, Altitudes = np.meshgrid(
        np.linspace(0, 0.9),
        np.linspace(0, 40000 * u.foot)
    )

    atmo = asb.Atmosphere(altitude=Altitudes.flatten())
    op_point = asb.OperatingPoint(
        atmosphere=atmo,
        velocity=Machs.flatten() * atmo.speed_of_sound()
    )
    sea_level_atmo = asb.Atmosphere()
    stagnation_density_ratio = (
            (op_point.total_pressure() / atmo.temperature()) /
            (sea_level_atmo.pressure() / sea_level_atmo.temperature())
    )

    Thrust_ratios = fit({
        "mach"                    : Machs.flatten(),
        "stagnation_density_ratio": stagnation_density_ratio,
    }).reshape(Machs.shape) / scalefactor

    Stagnation_density_ratio = stagnation_density_ratio.reshape(Machs.shape)

    p.contour(
        Machs, Altitudes / u.foot,
        Thrust_ratios,
        levels = np.arange(0, 1.51, 0.05),
        colorbar_label="Thrust ratio\n(Thrust) / (Sea-level Static Thrust)",
        alpha=0.5,
        linelabels_format=lambda x: f"{x:.2f}",
        cmap="RdBu_r"
    )
    plt.annotate(
        text="Data from Hamilton Sundstrand, AIAA 2003-6568",
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        ha="left",
        fontsize=9,
        alpha=0.5,
    )
    p.show_plot(
        "Effect of Transonic Mach and High Altitude\non Micro-Turbojet Dimensional Performance",
        "Mach [-]",
        "Altitude [foot]",
        dpi=300
    )

    fig, ax = plt.subplots()
    p.contour(
        Machs, Altitudes / u.foot,
        Stagnation_density_ratio,
        levels = np.arange(0, 1.51, 0.05),
        colorbar_label="Freestream Stagnation Density Ratio\n($\\rho_{t0}$) / (Sea-level Static $\\rho$)",
        alpha=0.5,
        linelabels_format=lambda x: f"{x:.2f}",
        cmap="RdBu_r",
        extend="both"
    )
    p.show_plot(
        "\nFreestream Stagnation Density Ratio",
        "Mach [-]",
        "Altitude [foot]",
        dpi=300
    )

    rel_eff = Thrust_ratios / Stagnation_density_ratio
    rel_eff /= np.max(rel_eff)

    fig, ax = plt.subplots()

    _, _, cbar= p.contour(
        Machs, Altitudes / u.foot,
        rel_eff,
        levels = np.arange(np.round(rel_eff.min(), 2), 1, 0.01),
        colorbar_label="Inlet Efficiency (Pressure Recovery)\n$\eta_i = P_{t2} / P_{t0}$",
        alpha=0.5,
        linelabels_format=lambda x: f"{x:.0%}",
        cmap="RdBu",
        extend="both"
    )
    plt.clim(vmax=1)
    cbar.ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, 0))
    p.set_ticks(0.1, 0.05, 5000, 1000)
    p.show_plot(
        "\nMicro-Turbojet Inferred Inlet Efficiency",
        "Mach [-]",
        "Altitude [foot]",
        dpi=300
    )