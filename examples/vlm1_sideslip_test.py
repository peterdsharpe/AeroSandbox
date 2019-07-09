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
a.set_paneling_everywhere(30, 30)
ap= vlm1(airplane=a,
         op_point = OperatingPoint(
    velocity = 10, alpha=15, beta = 15
))
ap.run()

ap.draw()

# Answer you should get: (XFLR5)
# CL = 0.869
# CDi = 0.064
# CL/CDi = 13.571
# Cm = -0.183