from classes import *

p = Airplane(
    name="Conventional",
    XYZle=[1, 0, 0],
    wings=[
        Wing(
            name="Main Wing"
            XYZle=[0, 0, 0],
            sections=[
                Wingsection( # Root
                    XYZle=[0, 0, 0],
                    chord=0.2,
                    twist=0,
                    airfoil=naca4412
                )

            ]
        )
    ],
)
