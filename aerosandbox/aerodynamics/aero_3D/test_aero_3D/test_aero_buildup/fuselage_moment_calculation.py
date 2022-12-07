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


# fuselage.draw()


def get_aero(xyz_ref):
    return asb.AeroBuildup(
        airplane=asb.Airplane(
            fuselages=[fuselage],
            xyz_ref=xyz_ref
        ),
        op_point=asb.OperatingPoint(
            velocity=10,
            alpha=5,
            beta=5
        )
    ).run()


x_cgs = np.linspace(0, 1, 11)
aeros = np.array([
    get_aero(xyz_ref=[x, 0, 0])
    for x in x_cgs
], dtype="O")

import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

fig, ax = plt.subplots(figsize=(4, 4))

plt.plot(
    x_cgs,
    np.array([a['m_b'] for a in aeros]),
    ".-"
)
p.show_plot(
    "Fuselage Pitching Moment",
    r"$x_{cg} / l$",
    "Pitching Moment [Nm]"
)
"""
Expected result:

For CG far forward (0), vehicle should have negative pitching moment.

For CG far aft (1), vehicle should have positive pitching moment.
"""
