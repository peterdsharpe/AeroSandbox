import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.tools.webplotdigitizer_reader import read_webplotdigitizer_csv
import pandas as pd
import aerosandbox.tools.units as u

raw_data = read_webplotdigitizer_csv("./wpd_datasets.csv")

data = {
    "mach": [],
    "altitude": [],
    "thrust": [],
}

for k, v in raw_data.items():
    N = len(v)
    data["mach"].append([float(k.split("M")[-1])] * N)
    data["altitude"].append(v[:, 0] * u.foot)
    data["thrust"].append(v[:, 1] * u.lbf)

df = pd.DataFrame({k: np.concatenate(v) for k, v in data.items()})

df["thrust_over_nominal_thrust"] = df["thrust"] / (120 * u.lbf)
df["density_ratio"] = df["altitude"].apply(
    func=lambda a: asb.Atmosphere(altitude=a).density() / asb.Atmosphere().density()
)


def get_stagnation_density_ratio(row):
    atmo = asb.Atmosphere(altitude=row["altitude"])
    op_point = asb.OperatingPoint(
        atmosphere=atmo, velocity=row["mach"] * atmo.speed_of_sound()
    )
    sea_level_atmo = asb.Atmosphere()
    return (op_point.total_pressure() / atmo.temperature()) / (
        sea_level_atmo.pressure() / sea_level_atmo.temperature()
    )


df["stagnation_density_ratio"] = df.apply(func=get_stagnation_density_ratio, axis=1)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    fig, ax = plt.subplots()
    plt.scatter(
        df["stagnation_density_ratio"],
        df["thrust_over_nominal_thrust"],
        s=10,
        c=df["mach"],
        cmap="rainbow",
    )
    plt.colorbar()
    plt.xscale("log")
    plt.yscale("log")
    p.show_plot(
        "",
        "Compressor-Face Stagnation Density ratio: $\\rho_\\mathrm{total} / \\rho_\\mathrm{sealevel}$",
        "Thrust ratio: $T / T_\\mathrm{nominal}$",
    )
