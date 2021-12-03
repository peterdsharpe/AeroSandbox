import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.aerodynamics.aero_3D.aero_buildup_submodels import fuselage_aerodynamics

from aerosandbox.tools.pretty_plots import plt, show_plot, contour, equal

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
Beta, Alpha = np.meshgrid(np.linspace(-90, 90, 500), np.linspace(-90, 90, 500))
aero = fuselage_aerodynamics(
    fuselage=fuselage,
    op_point=asb.OperatingPoint(
        velocity=10,
        alpha=Alpha,
        beta=Beta,
    )
)
from aerosandbox.tools.string_formatting import eng_string

contour(
    Beta, Alpha, aero["L"], colorbar_label="Lift $L$ [N]",
    # levels=100,
    linelabels_format=lambda s: eng_string(s, unit="N"),
    cmap=plt.get_cmap("coolwarm")
)
equal()
show_plot("3D Fuselage Lift", r"$\beta$ [deg]", r"$\alpha$ [deg]")
