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
            symmetric=False,
            xsecs=[
                asb.WingXSec(
                    xyz_le=[0, 0, 0],
                    chord=1,
                    twist=2,
                    airfoil=sd7037,
                ),
                asb.WingXSec(
                    xyz_le=[0, 0.5, 0],
                    chord=1,
                    twist=2,
                    airfoil=sd7037,
                ),
                asb.WingXSec(
                    xyz_le=[0.7, 4, 0],
                    chord=0.3,
                    twist=3,
                    airfoil=sd7037,
                )
            ]
        )
    ]
)

if __name__ == '__main__':
    airplane.draw_wireframe()
