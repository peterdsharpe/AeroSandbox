from get_data import turboprops, turbojets
import aerosandbox as asb
import aerosandbox.numpy as np
import pandas as pd
import aerosandbox.tools.units as u

df = pd.DataFrame({
    "OPR"        : turboprops["OPR"],
    "Weight [kg]": turboprops["Weight (dry) [lb]"] * u.lbm,
    # "Power [W]"  : turboprops["Power (TO) [shp]"] * u.hp,
    # "Thermal Efficiency [-]": 1 / (43.02e6 * turboprops["SFC (TO) [lb/shp hr]"] * (u.lbm / u.hp / u.hour)),
}).dropna()

targets = [
    "Power [W]",
    "Thermal Efficiency [-]",
]
inputs = [
    v for v in df.columns if v not in targets
]

##### Do Thermal Efficiency fit

target = "OPR"


def model(x, p):
    # return p["a"] * x ** p["Weight [kg]"]
    return np.blend(
        np.log10(x) / p["scl"] - p["cen"],
        value_switch_high=p["high"],
        value_switch_low=1,
    )

    # ideal_efficiency = 1 - (1 / x["OPR"]) ** ((1.4 - 1) / 1.4)
    #
    # res = np.blend(
    #     p["a"] + (np.log10(x["Weight [kg]"]) - p["wcen"]) / p["wscl"],
    #     value_switch_high=ideal_efficiency,
    #     value_switch_low=0,
    # )
    #
    # return res


fit = asb.FittedModel(
    model=model,
    x_data=df["Weight [kg]"].values,
    y_data=df[target].values,
    parameter_guesses={
        "scl" : 2,
        "cen" : 2,
        "high": 20,
    },
    parameter_bounds={
        "scl" : (0.1, 10),
        "cen" : (-2, 6),
        "high": (2, None),
    },
    # residual_norm_type="L1",
    # put_residuals_in_logspace=True,
    verbose=False,
    fit_type="upper bound"
)

print(f"Fit for {target}:")
print(fit.parameters)

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    fig, ax = plt.subplots()
    plt.scatter(
        df["Weight [kg]"],
        # df["Power [W]"],
        df["OPR"],
        alpha=0.4,
        label="Data"
    )
    x = np.geomspace(
        df["Weight [kg]"].min(),
        df["Weight [kg]"].max(),
        100
    )
    y = fit(x)
    plt.plot(
        x, y,
        alpha=0.7,
        label="Fit Model"
    )

    plt.xscale("log")
    # plt.yscale("log")
    plt.legend()
    p.show_plot(
        "Turboshaft: Weight vs. Technology-Limit OPR",
        "Weight [kg]",
        "Overall Pressure Ratio [-]",
    )
