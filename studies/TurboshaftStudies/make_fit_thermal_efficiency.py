from get_data import turboprops
import aerosandbox as asb
import aerosandbox.numpy as np
import pandas as pd
import aerosandbox.tools.units as u

df = pd.DataFrame({
    "OPR"                   : turboprops["OPR"],
    "Weight [kg]"           : turboprops["Weight (dry) [lb]"] * u.lbm,
    # "Power [W]"             : turboprops["Power (TO) [shp]"] * u.hp,
    "Thermal Efficiency [-]": 1 / (43.02e6 * turboprops["SFC (TO) [lb/shp hr]"] * (u.lbm / u.hp / u.hour)),
}).dropna()

targets = [
    "Power [W]",
    "Thermal Efficiency [-]",
]
inputs = [
    v for v in df.columns if v not in targets
]

##### Do Thermal Efficiency fit

target = "Thermal Efficiency [-]"


def model(x, p):
    ideal_efficiency = 1 - (1 / x["OPR"]) ** ((1.4 - 1) / 1.4)

    res = np.blend(
        p["a"] + (np.log10(x["Weight [kg]"]) - p["wcen"]) / p["wscl"],
        value_switch_high=ideal_efficiency,
        value_switch_low=0,
    )

    return res


fit = asb.FittedModel(
    model=model,
    x_data={
        v: df[v].values
        for v in inputs
    },
    y_data=df[target].values,
    parameter_guesses={
        "a"   : 0.2,
        "wcen": 3,
        "wscl": 7,
    },
    # residual_norm_type="L1",
    put_residuals_in_logspace=True,
    verbose=False

)

print(f"Fit for {target}:")
print(fit.parameters)

if __name__ == '__main__':

    df["knockdown"] = df["Thermal Efficiency [-]"] / (
            1 - (1 / df["OPR"]) ** ((1.4 - 1) / 1.4)
    )

    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    fig, ax = plt.subplots()
    plt.semilogx(
        df["Weight [kg]"] * np.random.uniform(0.9, 1.1, len(df)), # Add jitter
        df["knockdown"] * np.random.uniform(0.95, 1.05, len(df)), # Add jitter
        ".", alpha=0.4,
        label="Data"
    )
    x = np.geomspace(
        df["Weight [kg]"].min() / 3,
        df["Weight [kg]"].max() * 3,
        100
    )
    y = np.blend(
        fit.parameters["a"] + (np.log10(x) - fit.parameters["wcen"]) / fit.parameters["wscl"],
        value_switch_high=1,
        value_switch_low=0,
    )
    plt.plot(
        x, y,
        label="Model Fit",
        color="C0",
    )
    plt.annotate(
        text="Model is smoothly bounded by 0 and Brayton efficiency",
        xy=(0.98, 0.02),
        xycoords="axes fraction",
        ha="right",
        fontsize=9
    )

    p.show_plot(
        "Turboshaft: Thermal Efficiency Knockdown\nInputs: Weight",
        "Engine Weight [kg]",
        "Actual Efficiency relative to Ideal (Brayton) Efficiency\n$\eta_\mathrm{thermal}$ / $\eta_\mathrm{thermal,ideal}$",
    )
