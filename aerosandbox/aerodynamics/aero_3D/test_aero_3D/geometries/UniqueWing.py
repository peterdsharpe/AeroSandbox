import aerosandbox as asb
import aerosandbox.numpy as np

sd7037 = asb.Airfoil("naca4412", generate_polars=True)

airplane = asb.Airplane(
    name="Vanilla",
    xyz_ref=[0.5, 0, 0],
    wings=[
        asb.Wing(
            name="Wing",
            symmetric=False,
            xsecs=[
                asb.WingXSec(
                    xyz_le=[0, 0, 0],
                    chord=1,
                    twist=0,
                    airfoil=sd7037,
                ),
                asb.WingXSec(
                    xyz_le=[0, 0.5, 0],
                    chord=1,
                    twist=0,
                    airfoil=sd7037,
                ),
                asb.WingXSec(
                    xyz_le=[0, 4, 0],
                    chord=1,
                    twist=0,
                    airfoil=sd7037,
                )
            ]
        )
        # asb.Wing(
        #     name="H-stab",
        #     symmetric=False,
        #     xsecs=[
        #         asb.WingXSec(
        #             xyz_le=[0, 0, 0],
        #             chord=0.7,
        #         ),
        #         asb.WingXSec(
        #             xyz_le=[0.14, 1.25, 0],
        #             chord=0.42
        #         ),
        #     ]
        # ).translate([4, 0, 0]),
    ]
)

if __name__ == '__main__':
    airplane.draw_wireframe()