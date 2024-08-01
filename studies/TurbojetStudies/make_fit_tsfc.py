from get_data import turbojets
import aerosandbox as asb
import aerosandbox.numpy as np
import pandas as pd
import aerosandbox.tools.units as u

df = pd.DataFrame({
    "Weight [kg]" : turbojets["Dry Weight [lb]"] * u.lbm,
    "BPR"         : turbojets["BPR (static)"],
    "TSFC [kg/Ns]": turbojets["SFC (dry) [lb/lbf hr]"] * u.lbm / u.lbf / u.hour,
}).dropna()


##### Do TSFC fit


def model(x, p):
    return (
            p["a"]
            * x["Weight [kg]"] ** p["Weight [kg]"]
            # * x["Thrust [N]"] ** p["Thrust [N]"]
            * (x["BPR"] + p["BPR2"]) ** p["BPR"]
        # * np.exp(p["BPR"] * np.abs(x["BPR"] + 1e-100) ** p["BPR2"])
        # * x["OPR"] ** p["OPR"]
    )


fit = asb.FittedModel(
    model=model,
    x_data={
        "Weight [kg]": df["Weight [kg]"].values,
        "BPR"        : df["BPR"].values,
    },
    y_data=df["TSFC [kg/Ns]"].values,
    parameter_guesses={
        "a"          : 3e-5,
        "Weight [kg]": -0.1,
        "BPR"        : -0.3,
        "BPR2"       : 1,
    },
    parameter_bounds={
        "a"   : (0, None),
        "BPR2": (0, None),
    },
    # verbose=True,
    residual_norm_type="L1",
    put_residuals_in_logspace=True,
    verbose=False
)

print("Fit for TSFC:")
print(fit.parameters)
print(fit.goodness_of_fit())

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    fig, ax = plt.subplots()

    cmap = p.mpl.colormaps.get_cmap("turbo")
    norm = p.mpl.colors.Normalize(
        vmin=0,
        vmax=10
    )

    plt.scatter(
        df["Weight [kg]"],
        df["TSFC [kg/Ns]"],
        s=10,
        c=df["BPR"],
        cmap=cmap,
        norm=norm,
        alpha=0.6
    )

    for b in [0, 1, 3, 6, 10]:
        x = np.geomspace(
            df["Weight [kg]"].min(),
            df["Weight [kg]"].max(),
            100
        )
        y = fit({
            "Weight [kg]": x,
            "BPR"        : b
        })
        plt.plot(
            x, y,
            color=p.adjust_lightness(cmap(norm(b)), 0.9),
            label=f"BPR = {b}",
            alpha=0.7
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(1e-6, 1e-4)
    plt.colorbar(label="Bypass Ratio [-]")
    plt.legend(title="Model Fit", ncols=2, loc='lower left', fontsize=10)
    p.show_plot(
        "Turbojet: TSFC Model\nInputs: Weight, BPR",
        xlabel="Engine Weight [kg]",
        ylabel="Engine Thrust-Specific Fuel Consumption [kg/Ns]",
        legend=False
    )
