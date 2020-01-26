from aerosandbox import *

glider = Airplane(
    name="Conventional",
    xyz_ref=[0, 0, 0],
    wings=[
        Wing(
            name="Main Wing",
            xyz_le=[0, 0, 0],
            symmetric=False,
            xsecs=[
                WingXSec(
                    xyz_le=[0, 0, 0],
                    chord=1,
                    twist=0,
                    airfoil=Airfoil(name="naca0012"),
                    control_surface_type='symmetric',
                    control_surface_deflection=0,
                    control_surface_hinge_point=0
                ),
                WingXSec(
                    xyz_le=[0, 10, 0],
                    chord=1,
                    twist=0,
                    airfoil=Airfoil(name="naca0012"),
                    control_surface_type='symmetric',
                    control_surface_deflection=0,
                    control_surface_hinge_point=0
                )
            ]
        )
    ]
)

ap = vlm4(
    airplane=glider,
    op_point=OperatingPoint(
        velocity=10,
        alpha=0,
        beta=0,
        p=0,
        q=0,
        r=0,
    ),
)

ap.run()
ap.draw()

# Answer you should get: (XFLR5)
# CL = 0.797
# CDi = 0.017
# CL/CDi = 47.211
