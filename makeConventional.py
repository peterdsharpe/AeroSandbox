from classes import *

p = Airplane(
    name="Conventional",
    XYZref=[1, 0, 0],
    wings=[
        Wing(
            name="Main Wing",
            XYZle=[0, 0, 0],
            sections=[
                Wingsection(  # Root
                    XYZle=[0, 0, 0],
                    chord=0.2,
                    twist=0,
                    airfoil=Airfoil(name="naca4412")
                ),
                Wingsection(  # Mid
                    XYZle=[0.025, 0.4, 0],
                    chord=0.15,
                    twist=0,
                    airfoil=Airfoil(name="naca4412")
                ),
                Wingsection(  # Tip
                    XYZle=[0.075, 0.6, 0.1],
                    chord=0.05,
                    twist=0,
                    airfoil=Airfoil(name="naca4412")
                )
            ]
        ),
        Wing(
            name="Horizontal Stabilizer",
            XYZle=[1, 0, 0.2],
            sections=[
                Wingsection(  # root
                    XYZle=[0, 0, 0],
                    chord=0.1,
                    twist=0,
                    airfoil=Airfoil(name="naca0012")
                ),
                Wingsection(  # tip
                    XYZle=[0, 0.25, 0.15],
                    chord=0.1,
                    twist=0,
                    airfoil=Airfoil(name="naca0012")
                )
            ]
        )
    ]
)

