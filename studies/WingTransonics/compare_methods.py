import pandas as pd
import aerosandbox as asb
import aerosandbox.numpy as np
import json
from pathlib import Path

data_folder = Path(__file__).parent / "data" / "RAE2822_alpha_1deg_Re_6500000"


def get_data(filename):
    filename = Path(filename)

    if filename.suffix == ".json":
        with open(filename) as f:
            data = f.readlines()

        data = [
            json.loads(line)
            for line in data
        ]

        data = {
            k: np.array([d[k] for d in data])
            for k in data[0].keys()
        }
    elif filename.suffix == ".csv":
        data = pd.read_csv(filename)
        data = {
            k: np.array(data[k])
            for k in data.columns
        }

    order = np.argsort(data['mach'])
    indices = np.unique(data['mach'][order], return_index=True)[1]

    data = {
        k: v[order][indices]
        for k, v in data.items()
    }

    return data

def find_mach_dd(machs, CDs):
    from scipy import interpolate, optimize
    dCDdM = interpolate.PchipInterpolator(
        machs,
        CDs,
    ).derivative()

    return optimize.minimize_scalar(
        fun=lambda M: (dCDdM(M) - 0.1) ** 2,
        method="bounded",
        bounds=(0.5, 0.9)
    ).x


machs = np.linspace(0, 1.3, 500)
af = asb.Airfoil("rae2822")
ab_aero = af.get_aero_from_neuralfoil(
    1,
    Re=6.5e6,
    mach=machs,
    model_size="large"
)
ab_aero["mach"] = machs

datas = {
    "NeuralFoil \"large\""                       : ab_aero,
    "XFoil 6.98 (Potential Flow + IBL)": get_data(data_folder / "xfoil6.json"),
    "XFoil 7.02 (Full Potential + IBL)": get_data(data_folder / "xfoil7_viscous.csv"),
    "MSES (Euler + IBL)"               : get_data(data_folder / "mses.json"),
    "SU2 (RANS-SA + LM)"               : get_data(data_folder / "su2.json"),
}


import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

x = 'mach'

# for y in ['CL', 'CD']:
for y in ['CD']:
    fig, ax = plt.subplots(figsize=(6.5,4))

    fmts = [
        "o",
        "v",
        "s",
        "^",
        ">",
    ]

    for analysis, data in datas.items():
        plt.plot(
            data[x],
            data[y],
            "-" if "NeuralFoil" in analysis else fmts.pop(0),
            color="k" if "NeuralFoil" in analysis else None,
            label=analysis,
            markersize=4,
            markeredgewidth=0,
            alpha=0.7 if "NeuralFoil" in analysis else 0.9,
            zorder=5 if "NeuralFoil" in analysis else 4 if "XFoil" in analysis else 3 if "MSES" in analysis else 2 if "SU2" in analysis else 1
        )
    plt.xlim(0, 0.9)
    if y == 'CD':
        plt.ylim(0, 0.02)
        # plt.plot(
        #     machs,
        #     80 * np.maximum(machs - ab_aero["mach_crit"], 0) ** 4 + 0.0048,
        #     "k-",
        # )

    plt.legend(
        title="Analysis Method",
    )

    afax = ax.inset_axes([0.76, 0.802, 0.23, 0.23])
    afax.fill(
        af.x(), af.y(),
        facecolor=(0, 0, 0, 0.2), linewidth=1, edgecolor=(0, 0, 0, 0.7)
    )
    afax.annotate(
        text=f"{af.name.upper()} Airfoil",
        xy=(0.5, 0.11),
        xycoords="data",
        ha="center",
        va="bottom",
        fontsize=10,
        alpha=0.7
    )

    afax.grid(False)
    afax.set_xticks([])
    afax.set_yticks([])
    # afax.axis('off')
    afax.set_facecolor((1, 1, 1, 0.5))
    afax.set_xlim(-0.05, 1.05)
    afax.set_ylim(-0.12, 0.28)
    afax.set_aspect("equal", adjustable='box')

    p.show_plot(
        "Transonic Case: RAE2822 Airfoil, $\\alpha=1\\degree$, $\\mathrm{Re}=6.5\\times 10^6$",
        # x.capitalize(),
        # y,
        r"Mach Number $M_\infty$",
        r"Drag Coefficient $C_D$",
        legend=False,
        savefig="sample_validation_transonic.svg"
    )
