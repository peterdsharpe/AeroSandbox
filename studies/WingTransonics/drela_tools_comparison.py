import aerosandbox as asb
import aerosandbox.numpy as np

import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
from tqdm import tqdm

p.mpl.use('WebAgg')

fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=150)

airfoil = asb.Airfoil("rae2822")
Re = 1e6
alpha = 3

machs = np.concatenate([
    np.arange(0.05, 0.5, 0.05),
    np.arange(0.5, 0.6, 0.01),
    np.arange(0.6, 0.9, 0.005),
])

##### XFoil v6
xfoil6 = {}
for mach in tqdm(machs, desc="XFoil 6"):
    xfoil6[mach] = asb.XFoil(
        airfoil=airfoil,
        Re=Re,
        mach=mach,
        verbose=False,
    ).alpha(alpha)
xfoil6_Cds = {k: v['CD'] for k, v in xfoil6.items() if len(v['CD']) != 0}

plt.plot(
    np.array(list(xfoil6_Cds.keys()), dtype=float),
    np.concatenate(list(xfoil6_Cds.values())),
    ".-",
    label="XFoil 6"
)

##### XFoil v7
xfoil7 = {}
for mach in tqdm(machs, desc="XFoil 7"):
    try:
        xfoil7[mach] = asb.XFoil(
            airfoil=airfoil,
            Re=Re,
            mach=mach,
            xfoil_command="xfoil7",
            verbose=False
        ).alpha(alpha)
    except RuntimeError:
        pass
xfoil7_Cds = {k: v['CD'] for k, v in xfoil7.items() if len(v['CD']) != 0}

plt.plot(
    np.array(list(xfoil7_Cds.keys()), dtype=float),
    np.concatenate(list(xfoil7_Cds.values())),
    ".-",
    label="XFoil 7"
)

plt.ylim(bottom=0)
p.show_plot(
    f"Drela 2D Viscous Airfoil Tools Comparison\n{airfoil.name} Airfoil, alpha = {alpha}, Re = {Re}",
    "Mach [-]",
    "Drag Coefficient $C_D$"
)
