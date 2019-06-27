from AeroSandbox import *

a = Airplane(
        name="Conventional",
        xyz_ref=[0.05, 0, 0],
        wings=[
            Wing(
                name="Main Wing",
                xyz_le=[0, 0, 0],
                symmetric=True,
                sections=[
                    WingSection(  # Root
                        xyz_le=[0, 0, 0],
                        chord=0.18,
                        twist=2,
                        airfoil=Airfoil(name="naca4412")
                    ),
                    WingSection(  # Mid
                        xyz_le=[0.01, 0.5, 0],
                        chord=0.16,
                        twist=0,
                        airfoil=Airfoil(name="naca4412")
                    ),
                    WingSection(  # Tip
                        xyz_le=[0.08, 1, 0.1],
                        chord=0.08,
                        twist=-2,
                        airfoil=Airfoil(name="naca4412")
                    )
                ]
            ),
            Wing(
                name="Horizontal Stabilizer",
                xyz_le=[0.6, 0, 0.1],
                symmetric=True,
                sections=[
                    WingSection(  # root
                        xyz_le=[0, 0, 0],
                        chord=0.1,
                        twist=-10,
                        airfoil=Airfoil(name="naca0012")
                    ),
                    WingSection(  # tip
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
                sections=[
                    WingSection(
                        xyz_le=[0, 0, 0],
                        chord=0.1,
                        twist=0,
                        airfoil=Airfoil(name="naca0012")
                    ),
                    WingSection(
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

# a.set_vlm_paneling_everywhere(20,20)

ap = vlm1(
    airplane=a,
    op_point=OperatingPoint(velocity=10,
                            alpha=5,
                            beta=0),
)
ap.run()
ap.draw()

# Answer you should get: (XFLR5)
# CL = 0.797
# CDi = 0.017
# CL/CDi = 47.211
# Cm = -0.184