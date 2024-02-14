import aerosandbox.numpy as np
from aerosandbox.geometry.airplane import Airplane
from textwrap import indent, dedent
from aerosandbox.geometry.openvsp_io.asb_to_openvsp import _utilities
from aerosandbox.geometry.openvsp_io.asb_to_openvsp.wing_vspscript_generator import generate_wing
from aerosandbox.geometry.openvsp_io.asb_to_openvsp.fuselage_vspscript_generator import generate_fuselage
from aerosandbox.geometry.openvsp_io.asb_to_openvsp.propulsor_vspscript_generator import generate_propulsor


def generate_airplane(
        airplane: Airplane,
        include_main=True,
) -> str:
    """
    Generates a VSPScript file for an Airplane object.

    Args:
        airplane: The Airplane object to generate a VSPScript file for.

    Returns: A VSPScript file as a string.

    """
    script = ""

    for wing in airplane.wings:
        script += f"""\
{{
{indent(generate_wing(wing, include_main=False), "    ")}
}}
"""

    for fuselage in airplane.fuselages:
        script += f"""\
{{
{indent(generate_fuselage(fuselage, include_main=False), "    ")}
}}
"""

    for propulsor in airplane.propulsors:
        script += f"""\
{{
{indent(generate_propulsor(propulsor, include_main=False), "    ")}
}}
"""

    if include_main:
        script = _utilities.wrap_script(script)

    return script


if __name__ == '__main__':

    from aerosandbox.geometry.wing import Wing, WingXSec
    from aerosandbox.geometry.airfoil.airfoil import Airfoil

    wing = Wing(
        name="Main Wing",
        symmetric=True,
        xsecs=[
            WingXSec(
                xyz_le=[0.5, 0, 0],
                chord=1.1,
                twist=5,
                airfoil=Airfoil(name="dae11")
            ),
            WingXSec(
                xyz_le=[1, 2, 0],
                chord=0.9,
                twist=5,
                airfoil=Airfoil(name="NACA4412")
            ),
            WingXSec(
                xyz_le=[2, 5, 0],
                chord=0.5,
                twist=0,
                airfoil=Airfoil(name="NACA3412")
            ),
            WingXSec(
                xyz_le=[2.5, 5.5, 1],
                chord=0.25,
                twist=0,
                airfoil=Airfoil(name="NACA2412")
            )
        ]
    )

    from aerosandbox.geometry.fuselage import Fuselage, FuselageXSec

    af = Airfoil("ls013")
    x = np.sinspace(0, 1, 10)

    fuse = Fuselage(
        name="Fuse",
        xsecs=[
            FuselageXSec(
                xyz_c=[xi, 0, 0.05 * xi ** 2],
                width=2 * af.local_thickness(xi),
                height=af.local_thickness(xi),
                shape=4
            )
            for xi in x
        ]
    )

    from aerosandbox.geometry.propulsor import Propulsor

    prop = Propulsor(
        name="Prop",
        xyz_c=np.array([0.5, 0.4, 0.3]),
        xyz_normal=np.array([1, 0.2, 0.3]),
        radius=0.6,
    )

    airplane = Airplane(
        name="Aircraft",
        wings=[wing],
        fuselages=[fuse],
        propulsors=[prop]
    )

    print(generate_airplane(airplane))

    with open("test.vspscript", "w") as f:
        f.write(generate_airplane(airplane))
