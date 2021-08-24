import aerosandbox as asb
import aerosandbox.numpy as np

airfoil = asb.Airfoil("naca0008")

airplane = asb.Airplane(
    name="Flat Plate",
    xyz_ref=[0, 0, 0],
    wings=[
        asb.Wing(
            name="Wing",
            symmetric=False,
            xsecs=[
                asb.WingXSec(
                    xyz_le=[0, 0, 0],
                    chord=1,
                    twist=0,
                    airfoil=airfoil,
                ),
                asb.WingXSec(
                    xyz_le=[0, 10, 0],
                    chord=1,
                    twist=0,
                    airfoil=airfoil,
                ),
            ]
        )
    ]
)

if __name__ == '__main__':
    airplane.draw()
