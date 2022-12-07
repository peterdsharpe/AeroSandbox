import aerosandbox as asb
import aerosandbox.numpy as np

import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

fuselage = asb.Fuselage(
    xsecs=[
        asb.FuselageXSec(
            xyz_c=[xi, 0, 0],
            radius=asb.Airfoil("naca0020").local_thickness(xi) / 2
        )
        for xi in np.cosspace(0, 1, 20)
    ],
)

fig, ax = plt.subplots(figsize=(7, 6))
V = np.linspace(10, 1000, 1001)
op_point = asb.OperatingPoint(
    velocity=V,
)

aero = asb.AeroBuildup(
    airplane=asb.Airplane(
        fuselages=[fuselage]
    ),
    op_point=op_point,
).run()

plt.plot(
    op_point.mach(),
    aero["CD"],
    label="Full Model"
)

aero = asb.AeroBuildup(
    airplane=asb.Airplane(
        fuselages=[fuselage]
    ),
    op_point=op_point,
    include_wave_drag=False
).run()

plt.plot(
    op_point.mach(),
    aero["CD"],
    zorder=1.9,
    label="Model without Wave Drag"
)
p.show_plot(
    "Transonic Fuselage Drag",
    "Mach [-]",
    "Drag Area $C_D \cdot A$ [m$^2$]"
)

print("%.4g" % aero["CD"][-1])
