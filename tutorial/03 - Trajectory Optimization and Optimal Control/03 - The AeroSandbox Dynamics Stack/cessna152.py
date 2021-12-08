import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.tools import units as u


def ft_to_m(feet, inches=0):  # Converts feet (and inches) to meters
    return feet * u.foot + inches * u.inch


naca2412 = asb.Airfoil("naca2412")
naca0012 = asb.Airfoil("naca0012")
naca2412.generate_polars(cache_filename="assets/naca2412.json")
naca0012.generate_polars(cache_filename="assets/naca0012.json")

airplane = asb.Airplane(
    name="Cessna 152",
    wings=[
        asb.Wing(
            name="Wing",
            xsecs=[
                asb.WingXSec(
                    xyz_le=[0, 0, 0],
                    chord=ft_to_m(5, 4),
                    airfoil=naca2412
                ),
                asb.WingXSec(
                    xyz_le=[0, ft_to_m(7), ft_to_m(7) * np.sind(1)],
                    chord=ft_to_m(5, 4),
                    airfoil=naca2412
                ),
                asb.WingXSec(
                    xyz_le=[
                        ft_to_m(4, 3 / 4) - ft_to_m(3, 8 + 1 / 2),
                        ft_to_m(33, 4) / 2,
                        ft_to_m(33, 4) / 2 * np.sind(1)
                    ],
                    chord=ft_to_m(3, 8 + 1 / 2),
                    airfoil=naca0012
                )
            ],
            symmetric=True
        ),
        asb.Wing(
            name="Horizontal Stabilizer",
            xsecs=[
                asb.WingXSec(
                    xyz_le=[0, 0, 0],
                    chord=ft_to_m(3, 8),
                    airfoil=naca0012,
                    twist=-2
                ),
                asb.WingXSec(
                    xyz_le=[ft_to_m(1), ft_to_m(10) / 2, 0],
                    chord=ft_to_m(2, 4 + 3 / 8),
                    airfoil=naca0012,
                    twist=-2
                )
            ],
            symmetric=True
        ).translate([ft_to_m(13, 3), 0, ft_to_m(-2)]),
        asb.Wing(
            name="Vertical Stabilizer",
            xsecs=[
                asb.WingXSec(
                    xyz_le=[ft_to_m(-5), 0, 0],
                    chord=ft_to_m(8, 8),
                    airfoil=naca0012,
                ),
                asb.WingXSec(
                    xyz_le=[ft_to_m(0), 0, ft_to_m(1)],
                    chord=ft_to_m(3, 8),
                    airfoil=naca0012,
                ),
                asb.WingXSec(
                    xyz_le=[ft_to_m(0, 8), 0, ft_to_m(5)],
                    chord=ft_to_m(2, 8),
                    airfoil=naca0012,
                ),
            ]
        ).translate([ft_to_m(16, 11) - ft_to_m(3, 8), 0, ft_to_m(-2)])
    ],
    fuselages=[
        asb.Fuselage(
            xsecs=[
                asb.FuselageXSec(
                    xyz_c=[0, 0, ft_to_m(-1)],
                    radius=0,
                ),
                asb.FuselageXSec(
                    xyz_c=[0, 0, ft_to_m(-1)],
                    radius=ft_to_m(1.5)
                ),
                asb.FuselageXSec(
                    xyz_c=[ft_to_m(3), 0, ft_to_m(-0.85)],
                    radius=ft_to_m(1.7)
                ),
                asb.FuselageXSec(
                    xyz_c=[ft_to_m(5), 0, ft_to_m(0)],
                    radius=ft_to_m(2.7)
                ),
                asb.FuselageXSec(
                    xyz_c=[ft_to_m(10, 4), 0, ft_to_m(0.3)],
                    radius=ft_to_m(2.3)
                ),
                asb.FuselageXSec(
                    xyz_c=[ft_to_m(21, 11), 0, ft_to_m(0.8)],
                    radius=ft_to_m(0.3)
                ),
            ]
        ).translate([ft_to_m(-5), 0, ft_to_m(-3)])
    ]
)
