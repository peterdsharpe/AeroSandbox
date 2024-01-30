import aerosandbox as asb
import aerosandbox.numpy as np

import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

fuselage = asb.Fuselage(
    xsecs=[
        asb.FuselageXSec(
            xyz_c=[xi, 0, 0],
            radius=asb.Airfoil("naca0010").local_thickness(xi)
        )
        for xi in np.cosspace(0, 1, 20)
    ],
)

fig, ax = plt.subplots(figsize=(7, 6))
Beta, Alpha = np.meshgrid(
    np.linspace(-90, 90, 500),
    np.linspace(-90, 90, 500)
)
aero = asb.AeroBuildup(
    airplane=asb.Airplane(
        fuselages=[fuselage]
    ),
    op_point=asb.OperatingPoint(
        velocity=10,
        alpha=Alpha,
        beta=Beta,
    )
).run()

from aerosandbox.tools.string_formatting import eng_string

p.contour(
    Beta, Alpha, aero["L"], colorbar_label="Lift $L$ [N]",
    # levels=100,
    linelabels_format=lambda s: f"{s:.2g} N",
    cmap=p.mpl.colormaps.get_cmap("coolwarm")
)
p.equal()
plt.xlabel(r"$\beta$ [deg]")
plt.ylabel(r"$\alpha$ [deg]")
p.show_plot(
    "3D Fuselage Lift",
    rotate_axis_labels=False
)
