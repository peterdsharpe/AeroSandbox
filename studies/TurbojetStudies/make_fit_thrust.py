from get_data import turbojets, turbojets
import aerosandbox as asb
import aerosandbox.numpy as np
import pandas as pd
import aerosandbox.tools.units as u

df = pd.DataFrame({
    # "OPR"        : turbojets["OPR"],
    "Weight [kg]"   : turbojets["Dry Weight [lb]"] * u.lbm,
    "Dry Thrust [N]": turbojets["Thrust (dry) [lbf]"] * u.hp,
    # "Thermal Efficiency [-]": 1 / (43.02e6 * turbojets["SFC (TO) [lb/shp hr]"] * (u.lbm / u.hp / u.hour)),
}).dropna()

##### Do Power fit
target = "Thrust [N]"


def model(x, p):
    return (
            p["a"] * x ** p["w"]
    )


fit = asb.FittedModel(
    model=model,
    x_data=df["Weight [kg]"].values,
    y_data=df["Dry Thrust [N]"].values,
    parameter_guesses={
        "a": 1e4,
        "w": 1,
    },
    # residual_norm_type="L1",
    put_residuals_in_logspace=True,
    verbose=False,

)

print(f"Fit for {target}:")
print(fit.parameters)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    fig, ax = plt.subplots()

    plt.scatter(
        df["Weight [kg]"],
        df["Dry Thrust [N]"],
        alpha=0.4,
        s=10,
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
        label="Model Fit"
    )

    plt.xscale("log")
    plt.yscale("log")
    p.show_plot(
        "Turbojet: Thrust Model\nInputs: Weight",
        xlabel="Engine Weight [kg]",
        ylabel="Engine Dry Thrust [N]",
    )
