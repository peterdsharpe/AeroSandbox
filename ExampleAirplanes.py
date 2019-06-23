from Classes import *


def conventional():
    p = Airplane(
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
    p.set_ref_dims_from_wing()
    return p


def simple_airplane():
    # Reurns an airplane with a single, untwisted, untapered wing.
    p = Airplane(
        name="Single Wing",
        xyz_ref=[0, 0, 0],
        wings=[
            Wing(
                name="Wing",
                xyz_le=[0, 0, 0],
                symmetric=True,
                sections=[
                    WingSection(
                        xyz_le=[0, 0, 0],
                        chord=0.5,
                        twist=0,
                        airfoil=Airfoil(name="naca0012")
                    ),
                    WingSection(
                        xyz_le=[0, 1, 0],
                        chord=0.5,
                        twist=0,
                        airfoil=Airfoil(name="naca0012")
                    )
                ]
            )
        ]
    )
    p.set_ref_dims_from_wing()
    return p


def XFLR_default():
    p = Airplane(
        name="XFLR Default",
        xyz_ref=[0, 0, 0],
        wings=[
            Wing(
                name="Main Wing",
                xyz_le=[0, 0, 0],
                symmetric=True,
                chordwise_panels=13,
                sections=[
                    WingSection(  # Root
                        xyz_le=[0, 0, 0],
                        chord=0.18,
                        twist=0,
                        airfoil=Airfoil(name="naca0012"),
                        spanwise_panels=19,
                    ),
                    WingSection(  # Tip
                        xyz_le=[0.07, 1, 0],
                        chord=0.11,
                        twist=0,
                        airfoil=Airfoil(name="naca4412")
                    )
                ]
            ),
            Wing(
                name="Horizontal Stabilizer",
                xyz_le=[0.6, 0, 0.1],
                symmetric=True,
                chordwise_panels=7,
                sections=[
                    WingSection(  # root
                        xyz_le=[0, 0, 0],
                        chord=0.1,
                        twist=0,
                        airfoil=Airfoil(name="naca0012"),
                        spanwise_panels=7,
                    ),
                    WingSection(  # tip
                        xyz_le=[0.02, 0.17, 0],
                        chord=0.08,
                        twist=0,
                        airfoil=Airfoil(name="naca0012")
                    )
                ]
            ),
            Wing(
                name="Vertical Stabilizer",
                xyz_le=[0.65, 0, 0.1],
                symmetric=False,
                chordwise_panels=7,
                sections=[
                    WingSection(
                        xyz_le=[0, 0, 0],
                        chord=0.1,
                        twist=0,
                        airfoil=Airfoil(name="naca0012"),
                        spanwise_panels=7,
                    ),
                    WingSection(
                        xyz_le=[0.04, 0, 0.12],
                        chord=0.06,
                        twist=0,
                        airfoil=Airfoil(name="naca0012")
                    )
                ]
            )
        ]
    )
    p.set_ref_dims_from_wing()
    return p


def inverted_T_test():
    p = Airplane(
        name="Inverted T Test",
        xyz_ref=[0, 0, 0],
        wings=[
            Wing(
                name="Horizontal Part",
                xyz_le=[0, 0, 0],
                symmetric=True,
                sections=[
                    WingSection(
                        xyz_le=[0, 0, 0],
                        chord=0.1,
                    ),
                    WingSection(
                        xyz_le=[0,0.5,0],
                        chord=0.1,
                    )
                ]
            ),
            Wing(
                name="Vertical Part",
                xyz_le=[0,0,0],
                symmetric=False,
                sections=[
                    WingSection(
                        xyz_le=[0,0,0],
                        chord=0.1,
                    ),
                    WingSection(
                        xyz_le=[0,0.5,0],
                        chord=0.1,
                    )
                ]
            )
        ]
    )
    p.set_ref_dims_from_wing()
    return p
