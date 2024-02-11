import aerosandbox.numpy as np
from aerosandbox.geometry.propulsor import Propulsor
from textwrap import indent, dedent
from aerosandbox.geometry.openvsp_io.asb_to_openvsp import _utilities


def generate_propulsor(propulsor: Propulsor, include_main=True) -> str:
    """
    Generates a VSPScript file for a Wing object.

    Args:
        wing: The Wing object to generate a VSPScript file for.

    Returns: A VSPScript file as a string.

    """
    script = ""

    script += f"""\
//==== Add Propulsor "{propulsor.name}" ====//
string pid = AddGeom("PROP");

"""

    script += f"""\
//==== Set Overall Propulsor Options and First Section ====//
SetParmVal( pid, "X_Rel_Location", "XForm", {propulsor.xyz_c[0]} );
SetParmVal( pid, "Y_Rel_Location", "XForm", {propulsor.xyz_c[1]} );
SetParmVal( pid, "Z_Rel_Location", "XForm", {propulsor.xyz_c[2]} );
SetParmVal( pid, "Diameter", "Design", {2 * propulsor.radius} );
SetParmVal( pid, "PropMode", "Design", 1.0 ); // 0 = Blades, 1 = Both, 2 = Disk

Update();

"""

    script += """\
SetParmVal( pid, "Tess_U", "Shape", 25 );
SetParmVal( pid, "Tess_W", "Shape", 100 );
Update();
    """

    if include_main:
        script = _utilities.wrap_script(script)

    return script


if __name__ == '__main__':

    prop = Propulsor(
        name="Prop",
        xyz_c=np.array([0.5, 0.4, 0.3]),
        xyz_normal=np.array([1, 0.2, 0.3]),
        radius=0.6,
    )
    print(generate_propulsor(prop))

    with open("test.vspscript", "w") as f:
        f.write(generate_propulsor(prop))
