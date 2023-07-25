import aerosandbox as asb
import aerosandbox.numpy as np

sd7037 = asb.Airfoil("sd7037")

airplane = asb.Airplane(
    name="Vanilla",
    xyz_ref=[0.5, 0, 0],
    s_ref=9,
    c_ref=0.9,
    b_ref=10,
    wings=[
        asb.Wing(
            name="Wing",
            symmetric=True,
            xsecs=[
                asb.WingXSec(
                    xyz_le=[0, 0, 0],
                    chord=1,
                    twist=2,
                    airfoil=sd7037,
                ),
                asb.WingXSec(
                    xyz_le=[0.2, 5, 1],
                    chord=0.6,
                    twist=2,
                    airfoil=sd7037,
                )
            ]
        ),
        asb.Wing(
            name="H-stab",
            symmetric=True,
            xsecs=[
                asb.WingXSec(
                    xyz_le=[0, 0, 0],
                    chord=0.7,
                    airfoil=asb.Airfoil("naca0012")
                ),
                asb.WingXSec(
                    xyz_le=[0.14, 1.25, 0],
                    chord=0.42,
                    airfoil=asb.Airfoil("naca0012")
                ),
            ]
        ).translate([4, 0, 0]),
        asb.Wing(
            name="V-stab",
            xsecs=[
                asb.WingXSec(
                    xyz_le=[0, 0, 0],
                    chord=0.7,
                    airfoil=asb.Airfoil("naca0012")
                ),
                asb.WingXSec(
                    xyz_le=[0.14, 0, 1],
                    chord=0.42,
                    airfoil=asb.Airfoil("naca0012")
                )
            ]
        ).translate([4, 0, 0])
    ],
    fuselages=[
        asb.Fuselage(
            name="Fuselage",
            xsecs=[
                asb.FuselageXSec(
                    xyz_c=[xi * 5 - 0.5, 0, 0],
                    radius=asb.Airfoil("naca0024").local_thickness(x_over_c=xi)
                )
                for xi in np.cosspace(0, 1, 30)
            ]
        )
    ]
)

if __name__ == '__main__':
    airplane.draw()
