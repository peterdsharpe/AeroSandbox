import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.tools import units as u


def ft(feet, inches=0.):  # Converts feet (and inches) to meters
    return feet * u.foot + inches * u.inch


airplane = asb.Airplane(
    name="Cessna 152",
    wings=[
        asb.Wing(
            name="Wing",
            xsecs=[
                asb.WingXSec(
                    xyz_le=[0, 0, 0],
                    chord=ft(5, 4),
                    airfoil=asb.Airfoil("naca2412")
                ),
                asb.WingXSec(
                    xyz_le=[0, ft(7), ft(7) * np.sind(1)],
                    chord=ft(5, 4),
                    airfoil=asb.Airfoil("naca2412")
                ),
                asb.WingXSec(
                    xyz_le=[
                        ft(4, 3 / 4) - ft(3, 8 + 1 / 2),
                        ft(33, 4) / 2,
                        ft(33, 4) / 2 * np.sind(1)
                    ],
                    chord=ft(3, 8 + 1 / 2),
                    airfoil=asb.Airfoil("naca0012")
                )
            ],
            symmetric=True
        ),
        asb.Wing(
            name="Horizontal Stabilizer",
            xsecs=[
                asb.WingXSec(
                    xyz_le=[0, 0, 0],
                    chord=ft(3, 8),
                    airfoil=asb.Airfoil("naca0012"),
                    twist=-2
                ),
                asb.WingXSec(
                    xyz_le=[ft(1), ft(10) / 2, 0],
                    chord=ft(2, 4 + 3 / 8),
                    airfoil=asb.Airfoil("naca0012"),
                    twist=-2
                )
            ],
            symmetric=True
        ).translate([ft(13, 3), 0, ft(-2)]),
        asb.Wing(
            name="Vertical Stabilizer",
            xsecs=[
                asb.WingXSec(
                    xyz_le=[ft(-5), 0, 0],
                    chord=ft(8, 8),
                    airfoil=asb.Airfoil("naca0012"),
                ),
                asb.WingXSec(
                    xyz_le=[ft(0), 0, ft(1)],
                    chord=ft(3, 8),
                    airfoil=asb.Airfoil("naca0012"),
                ),
                asb.WingXSec(
                    xyz_le=[ft(0, 8), 0, ft(5)],
                    chord=ft(2, 8),
                    airfoil=asb.Airfoil("naca0012"),
                ),
            ]
        ).translate([ft(16, 11) - ft(3, 8), 0, ft(-2)])
    ],
    fuselages=[
        asb.Fuselage(
            xsecs=[
                asb.FuselageXSec(
                    xyz_c=[0, 0, ft(-1)],
                    radius=0,
                ),
                asb.FuselageXSec(
                    xyz_c=[0, 0, ft(-1)],
                    radius=ft(1.5),
                    shape=4  # Create a superellipse with shape parameter 4.
                ),
                asb.FuselageXSec(
                    xyz_c=[ft(3), 0, ft(-0.85)],
                    radius=ft(1.7),
                    shape=4
                ),
                asb.FuselageXSec(
                    xyz_c=[ft(5), 0, ft(0)],
                    radius=ft(2.7),
                    shape=3
                ),
                asb.FuselageXSec(
                    xyz_c=[ft(10, 4), 0, ft(0.3)],
                    radius=ft(2.3),
                    shape=3
                ),
                asb.FuselageXSec(
                    xyz_c=[ft(21, 11), 0, ft(0.8)],
                    radius=ft(0.3)
                ),
            ]
        ).translate([ft(-5), 0, ft(-3)]).subdivide_sections(2)
    ]
)

if __name__ == '__main__':
    airplane.draw_three_view()
