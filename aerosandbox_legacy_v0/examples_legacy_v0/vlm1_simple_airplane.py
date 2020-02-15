from aerosandbox import *

a = Airplane(
        name="Single Wing",
        xyz_ref=[0, 0, 0],
        wings=[
            Wing(
                name="Wing",
                xyz_le=[0, 0, 0],
                symmetric=True,
                xsecs=[
                    WingXSec(
                        xyz_le=[0, 0, 0],
                        chord=0.5,
                        twist=0,
                        airfoil=Airfoil(name="naca0012")
                    ),
                    WingXSec(
                        xyz_le=[0, 1, 0],
                        chord=0.5,
                        twist=0,
                        airfoil=Airfoil(name="naca0012")
                    )
                ]
            )
        ]
    )
a.set_ref_dims_from_wing()

ap = vlm1(
    airplane=a,
    op_point=OperatingPoint(velocity=10,
                            alpha=5,
                            beta=0),
)
ap.run()
ap.draw()

# Answer you should get: (XFLR5)
# CL = 0.320
# CDi = 0.008
# CL/CDi = 40.157
# Cm = -0.074