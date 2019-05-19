from classes import *

p = Airplane(
    name="Conventional",
    xyz_ref=[0.05, 0, 0],
    wings=[
        Wing(
            name="Main Wing",
            xyz_le=[0, 0, 0],
            sections=[
                WingSection(  # Root
                    xyz_le=[0, 0, 0],
                    chord=0.2,
                    twist=0,
                    airfoil=Airfoil(name="naca4412")
                ),
                WingSection(  # Mid
                    xyz_le=[0.025, 0.4, 0],
                    chord=0.15,
                    twist=0,
                    airfoil=Airfoil(name="naca4412")
                ),
                WingSection(  # Tip
                    xyz_le=[0.075, 0.6, 0.1],
                    chord=0.05,
                    twist=0,
                    airfoil=Airfoil(name="naca4412")
                )
            ]
        ),
        Wing(
            name="Horizontal Stabilizer",
            xyz_le=[1, 0, 0.2],
            sections=[
                WingSection(  # root
                    xyz_le=[0, 0, 0],
                    chord=0.1,
                    twist=0,
                    airfoil=Airfoil(name="naca0012")
                ),
                WingSection(  # tip
                    xyz_le=[0, 0.25, 0.15],
                    chord=0.1,
                    twist=0,
                    airfoil=Airfoil(name="naca0012")
                )
            ]
        )
    ]
)

p.plot_geometry()

print("done now")