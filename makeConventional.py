from classes import *

p = Airplane(
    name="Conventional",
    XYZle=[1, 0, 0],
    wings=[
        Wing(
            name="Main Wing",
            XYZle=[0, 0, 0],
            sections=[
                Wingsection(  # Root
                    XYZle=[0, 0, 0],
                    chord=0.2,
                    twist=0,
                    airfoil=Airfoil("naca4412")
                ),
                Wingsection(  # Mid
                    XYZle=[0.025, 0.4, 0],
                    chord=0.15,
                    twist=0,
                    airfoil=Airfoil("naca4412")
                ),
                Wingsection(  # Tip
                    XYZle=[0.075, 0.6, 0.1],
                    chord=0.05,
                    twist=0,
                    airfoil=Airfoil("naca4412")
                )
            ]
        )
    ],
)
