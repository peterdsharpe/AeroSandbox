import aerosandbox.numpy as np
from aerosandbox.geometry.propulsor import Propulsor
from aerosandbox.geometry.openvsp_io.asb_to_openvsp import _utilities


def generate_propulsor(propulsor: Propulsor, include_main=True) -> str:
    """
    Generate a VSPScript file for a Propulsor object.

    Parameters
    ----------
    propulsor : Propulsor
        The Propulsor object to generate a VSPScript file for.
    include_main
        If True, wraps the script in a main() function via `_utilities.wrap_script()`.

    Returns
    -------
    str
        A VSPScript file as a string.
    """
    script = ""

    script += f"""\
//==== Add Propulsor "{propulsor.name}" ====//
string pid = AddGeom("PROP");

"""

    ### Compute the orientation of the propulsor
    desired_normal = np.array(propulsor.xyz_normal) / np.linalg.norm(
        propulsor.xyz_normal
    )

    if np.allclose(desired_normal, np.array([1, 0, 0]), atol=1e-8):
        y_rot_deg = 180
        z_rot_deg = 0
    else:
        ### Solve, in closed form, for the rotation angles that align the
        ### propulsor's default normal [-1, 0, 0] with the desired normal.
        # The rotation `R_y(y_rot) @ R_z(z_rot)` maps [-1, 0, 0] to:
        #   [-cos(y_rot) * cos(z_rot), -sin(z_rot), sin(y_rot) * cos(z_rot)]
        # Setting this equal to `desired_normal` = [nx, ny, nz] and solving
        # gives the expressions below. (Since z_rot is restricted to
        # [-90, 90] deg, cos(z_rot) >= 0, which makes the atan2 form valid.)
        nx, ny, nz = desired_normal
        # `+ 0.` canonicalizes -0.0 to 0.0
        z_rot_deg = np.degrees(-np.arcsin(np.clip(ny, -1, 1))) + 0.0
        y_rot_deg = np.degrees(np.arctan2(nz, -nx)) + 0.0

    script += f"""\
//==== Set Overall Propulsor Options and First Section ====//
SetParmVal( pid, "Y_Rel_Rotation", "XForm", {y_rot_deg:.8g} );
SetParmVal( pid, "Z_Rel_Rotation", "XForm", {z_rot_deg:.8g} );
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


if __name__ == "__main__":
    prop = Propulsor(
        name="Prop",
        xyz_c=np.array([0, 0, 0]),
        xyz_normal=np.array([1, 0.1, 0]),
        radius=0.6,
    )
    print(generate_propulsor(prop))

    with open("test.vspscript", "w") as f:
        f.write(generate_propulsor(prop))
