import aerosandbox as asb
import aerosandbox.numpy as np

sd7037 = asb.Airfoil("sd7037")

sd7037.generate_polars(cache_filename="cache/sd7037")

airplane = asb.Airplane(
    name="Vanilla",
    xyz_ref=[0.5, 0, 0],
    wings=[
        asb.Wing(
            name="Wing",
            symmetric=True,
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
                    xyz_le=[0.5, 4, 1],
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
