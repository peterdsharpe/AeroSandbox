from aerosandbox import *

p = Airplane(
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
                    airfoil=Airfoil(name="naca9412")
                ),
                WingXSec(
                    xyz_le=[0, 1, 0],
                    chord=0.5,
                    twist=-5,
                    airfoil=Airfoil(name="naca9412")
                )
            ]
        )
    ]
)
p.set_ref_dims_from_wing()
p.set_paneling_everywhere(30, 30)
ap= vlm1(airplane=p,
         op_point = OperatingPoint(
    velocity = 10, alpha=0, beta = 0
))
ap.run()
ap.draw()

# Answer you should get: (XFLR5)
# CL = 0.471
# CDi = 0.017
# CL/CDi = 27.065
# Cm = -0.338