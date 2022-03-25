import pandas as pd
import aerosandbox as asb
import aerosandbox.numpy as np
import json
from pathlib import Path

data_folder = Path(__file__).parent / "data" / "RAE2822_alpha_1deg_Re_6500000"

def get_data(filename):
    with open(filename) as f:
        data= f.readlines()

    data = [
        json.loads(line)
        for line in data
    ]

    data = {
        k: np.array([d[k] for d in data])
        for k in data[0].keys()
    }

    order = np.argsort(data['mach'])

    data = {
        k: v[order]
        for k, v in data.items()
    }

    return data

machs = np.linspace(0, 1.2, 500)
airfoil = asb.Airfoil("rae2822")
airfoil.generate_polars(cache_filename="./cache/rae2822.json")
aerobuildup_data={
    "mach": machs,
    "CL": airfoil.CL_function(1, 6.5e6, machs, 0) * np.ones_like(machs),
    "CD": airfoil.CD_function(1, 6.5e6, machs, 0) * np.ones_like(machs),
    "CM": airfoil.CM_function(1, 6.5e6, machs, 0) * np.ones_like(machs),
}

datas = {
    "XFoil v6": get_data(data_folder / "xfoil6.csv"),
    "MSES": get_data(data_folder / "mses.csv"),
    "SU2": get_data(data_folder / "su2.csv"),
    "ASB AeroBuildup": aerobuildup_data,
}

import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

fig, ax = plt.subplots()

x = 'mach'
y = 'CD'

for analysis, data in datas.items():


    plt.plot(
        data[x],
        data[y],
        "-",
        label=analysis,
        # linewidth=1,
        # markersize=3,
        alpha=0.8
    )

# plt.xlim(left=0)
# plt.ylim(bottom=0)
# plt.xlim(0.7, 0.95)
# plt.xlim(0.6, 0.8)
# plt.ylim(0, 1)
plt.ylim(0, 0.02)
# p.set_ticks(0.1, 0.02)
p.show_plot(
    "Comparison of Aerodynamic Analysis Methods\nRAE2822 Airfoil, AoA=1 deg, Re=6.5M",
    x,
    y
)