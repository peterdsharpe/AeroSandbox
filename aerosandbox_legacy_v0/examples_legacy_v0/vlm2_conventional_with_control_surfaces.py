from aerosandbox import *

a = Airplane(
        name="Conventional",
        xyz_ref=[0.0, 0, 0],
        wings=[
            Wing(
                name="Main Wing",
                xyz_le=[0, 0, 0],
                symmetric=True,
                xsecs=[
                    WingXSec(  # Root
                        xyz_le=[0, 0, 0],
                        chord=0.18,
                        twist=2,
                        airfoil=Airfoil(name="naca4412"),
                        control_surface_type='symmetric',
                        control_surface_deflection=0,
                        control_surface_hinge_point=0.75
                    ),
                    WingXSec(  # Mid
                        xyz_le=[0.01, 0.5, 0],
                        chord=0.16,
                        twist=0,
                        airfoil=Airfoil(name="naca4412"),
                        control_surface_type='asymmetric',
                        control_surface_deflection=0,
                        control_surface_hinge_point=0.75
                    ),
                    WingXSec(  # Tip
                        xyz_le=[0.08, 1, 0.1],
                        chord=0.08,
                        twist=-2,
                        airfoil=Airfoil(name="naca4412"),
                    )
                ]
            ),
            Wing(
                name="Horizontal Stabilizer",
                xyz_le=[0.6, 0, 0.1],
                symmetric=True,
                xsecs=[
                    WingXSec(  # root
                        xyz_le=[0, 0, 0],
                        chord=0.1,
                        twist=-10,
                        airfoil=Airfoil(name="naca0012"),
                        control_surface_type='symmetric',
                        control_surface_deflection=0,
                        control_surface_hinge_point=0.75
                    ),
                    WingXSec(  # tip
                        xyz_le=[0.02, 0.17, 0],
                        chord=0.08,
                        twist=-10,
                        airfoil=Airfoil(name="naca0012")
                    )
                ]
            ),
            Wing(
                name="Vertical Stabilizer",
                xyz_le=[0.6, 0, 0.15],
                symmetric=False,
                xsecs=[
                    WingXSec(
                        xyz_le=[0, 0, 0],
                        chord=0.1,
                        twist=0,
                        airfoil=Airfoil(name="naca0012"),
                        control_surface_type='symmetric',
                        control_surface_deflection=0,
                        control_surface_hinge_point=0.75
                    ),
                    WingXSec(
                        xyz_le=[0.04, 0, 0.15],
                        chord=0.06,
                        twist=0,
                        airfoil=Airfoil(name="naca0012")
                    )
                ]
            )
        ]
    )
a.set_ref_dims_from_wing()

ap = vlm2(
    airplane=a,
    op_point=OperatingPoint(velocity=10,
                            alpha=5,
                            beta=0),
)
ap.run()

# Answer you should get: (XFLR5)
# CL = 0.797
# CDi = 0.017
# CL/CDi = 47.211
# Cm = -0.184