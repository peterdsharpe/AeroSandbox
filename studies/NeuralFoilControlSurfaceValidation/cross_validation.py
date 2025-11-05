import aerosandbox as asb
import aerosandbox.numpy as np

af = asb.Airfoil("naca0012")
Re = 1e6
mach = 0.1

control_surface = asb.ControlSurface(deflection=1, hinge_point=0.75)

af_deflected = af.add_control_surface(
    deflection=1,
    hinge_point_x=0.75,
)
alpharange = np.linspace(-5, 5, 31)

aeros = {}

aeros["XFoil Undeflected"] = asb.XFoil(
    airfoil=af, Re=Re, mach=mach, max_iter=100, verbose=True
).alpha(alpharange)
aeros["XFoil Deflected"] = asb.XFoil(
    airfoil=af_deflected, Re=Re, mach=mach, max_iter=100, verbose=True
).alpha(alpharange)
aeros["NeuralFoil Undeflected"] = af.get_aero_from_neuralfoil(
    alpha=alpharange,
    Re=Re,
    mach=mach,
    model_size="xxxlarge",
)
aeros["NeuralFoil Undeflected"]["alpha"] = alpharange
aeros["NeuralFoil Deflected"] = af_deflected.get_aero_from_neuralfoil(
    alpha=alpharange,
    Re=Re,
    mach=mach,
    model_size="xxxlarge",
)
aeros["NeuralFoil Deflected"]["alpha"] = alpharange

import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

fig, ax = plt.subplots(1, 3, sharex=True)

for i, var in enumerate(["CL", "CD", "CM"]):
    for label, aero in aeros.items():
        ax[i].plot(
            aero["alpha"],
            aero[var],
            label=label + " " + var,
            color="C0" if "XFoil" in label else "C1",
            linestyle="-" if "Undeflected" in label else "--",
            alpha=0.5,
        )

p.show_plot(
    "Control Surface Validation",
    xlabel="Angle of Attack [deg]",
)
