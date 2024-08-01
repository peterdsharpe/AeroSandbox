import aerosandbox as asb
import aerosandbox.tools.units as u
import pandas as pd

data_file = asb._asb_root / "library" / "datasets" / "turbine_engines" / "data.xlsx"

data = pd.read_excel(data_file, sheet_name=None)

data["Civilian Turboprops"]["is_military"] = False
data["Military Turboprops"]["is_military"] = True
data["Civilian Turbojets"]["is_military"] = False
data["Military Turbojets"]["is_military"] = True

turboprops = pd.concat([
    data["Civilian Turboprops"],
    data["Military Turboprops"],
])
turbojets = pd.concat([
    data["Civilian Turbojets"],
    data["Military Turbojets"],
])

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    fig, ax = plt.subplots()
    plt.plot(
        turboprops["Power (TO) [shp]"] * u.hp,
        turboprops["Weight (dry) [lb]"] * u.lbm,
        ".",
        label="Turboprops",
        alpha=0.3
    )
    plt.xscale('log')
    plt.yscale('log')
    p.show_plot(
        "",
        "Power [W]",
        "Weight [kg]",
    )

    fig, ax = plt.subplots()
    plt.plot(
        turboprops["Power (TO) [shp]"] * u.hp,
        1 / (43.02e6 * turboprops["SFC (TO) [lb/shp hr]"] * (u.lbm / u.hp / u.hour)),
        ".",
        label="Turboprops",
        alpha=0.3
    )
    plt.xscale('log')
    ax.yaxis.set_major_formatter(p.ticker.PercentFormatter(xmax=1))
    p.show_plot(
        "",
        "Power [W]",
        "Thermal Efficiency [%]",
    )

    # fig, ax = plt.subplots()
    # p.sns.pairplot(data["civilian_turboprops"])
