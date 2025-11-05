from get_data import turboprops
import aerosandbox as asb
import aerosandbox.numpy as np
import pandas as pd
import aerosandbox.tools.units as u

df = pd.DataFrame(
    {
        "OPR": turboprops["OPR"],
        "Weight [kg]": turboprops["Weight (dry) [lb]"] * u.lbm,
        "Power [W]": turboprops["Power (TO) [shp]"] * u.hp,
        # "Thermal Efficiency [-]": 1 / (43.02e6 * turboprops["SFC (TO) [lb/shp hr]"] * (u.lbm / u.hp / u.hour)),
    }
).dropna()

targets = [
    "Power [W]",
    "Thermal Efficiency [-]",
]
inputs = [v for v in df.columns if v not in targets]

##### Do Power fit
target = "Power [W]"


def model(x, p):
    res = p["a"]
    for v in inputs:
        res *= x[v] ** p[v]

    return res


fit = asb.FittedModel(
    model=model,
    x_data={v: df[v].values for v in inputs},
    y_data=df[target].values,
    parameter_guesses={**{"a": 1}, **{v: 1 for v in inputs}},
    residual_norm_type="L1",
    put_residuals_in_logspace=True,
    verbose=False,
)

print(f"Fit for {target}:")
print(fit.parameters)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    fig, ax = plt.subplots()

    cmap = p.mpl.colormaps.get_cmap("turbo")
    norm = p.mpl.colors.Normalize(vmin=df["OPR"].min(), vmax=df["OPR"].max())

    plt.scatter(
        df["Weight [kg]"] * np.random.uniform(0.95, 1.05, size=len(df)),
        df["Power [W]"] * np.random.uniform(0.95, 1.05, size=len(df)),
        s=10,
        c=df["OPR"],
        cmap=cmap,
        norm=norm,
        alpha=0.6,
    )

    for o in [5, 10, 15, 20]:
        x = np.geomspace(df["Weight [kg]"].min(), df["Weight [kg]"].max(), 100)
        y = fit({"Weight [kg]": x, "OPR": o})
        plt.plot(
            x,
            y,
            color=p.adjust_lightness(cmap(norm(o)), 0.9),
            label=f"OPR = {o}",
            alpha=0.7,
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.colorbar(label="Overall Pressure Ratio [-]")
    plt.legend(title="Model Fit")
    p.show_plot(
        "Turboshaft: Power Model\nInputs: Weight, OPR",
        xlabel="Engine Weight [kg]",
        ylabel="Engine Rated Power [W]",
        legend=False,
    )
