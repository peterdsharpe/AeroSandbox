import aerosandbox.numpy as np
from aerosandbox.geometry.fuselage import Fuselage, FuselageXSec
from textwrap import indent, dedent
from aerosandbox.geometry.openvsp_io.asb_to_openvsp import _utilities


def generate_fuselage(
        fuselage: Fuselage,
        include_main=True,
        continuity_type: str = "C2",
) -> str:
    """
    Generates a VSPScript file for a Fuselage object.

    Args:
        fuselage: The Fuselage object to generate a VSPScript file for.

    Returns: A VSPScript file as a string.

    """
    script = ""

    script += f"""\
//==== Add Fuselage "{fuselage.name}" ====//
string fid = AddGeom("FUSELAGE");
ChangeXSecShape( GetXSecSurf( fid, 0 ), 0, XS_SUPER_ELLIPSE );
ChangeXSecShape( GetXSecSurf( fid, 4 ), 4, XS_SUPER_ELLIPSE );

"""

    script += """//==== Generate Blank Fuselage Sections ====//\n"""
    if len(fuselage.xsecs) < 2:
        raise ValueError("Fuselages must have at least 2 cross sections.")
    else:
        script += f"""\
SetParmVal( fid, "XLocPercent", "XSec_3", 1.0 ); // Causes all sections to be generated at x/L = 1, so they don't conflict
        """
        for i in range(len(fuselage.xsecs) - 1):
            script += f"""\
InsertXSec( fid, 3, XS_SUPER_ELLIPSE ); // FuselageXSecs {i} to {i + 1}
"""

    length = fuselage.xsecs[-1].xyz_c[0] - fuselage.xsecs[0].xyz_c[0]
    script += f"""\
//===== Cut The Original Section =====//
CutXSec( fid, 0 );
CutXSec( fid, 0 );
CutXSec( fid, 0 );
CutXSec( fid, 0 );

//==== Set Overall Fuselage Options and First Section ====//
SetParmVal( fid, "X_Rel_Location", "XForm", {fuselage.xsecs[0].xyz_c[0]} );
SetParmVal( fid, "Y_Rel_Location", "XForm", {fuselage.xsecs[0].xyz_c[1]} );
SetParmVal( fid, "Z_Rel_Location", "XForm", {fuselage.xsecs[0].xyz_c[2]} );
SetParmVal( fid, "Length", "Design", {length} );
Update();

"""
    # TODO symmetry here

    script += """//==== Set Fuselage Section Options ====//\n"""
    if continuity_type == "C0":
        continuity_type_string = "0.0"
    elif continuity_type == "C1":
        continuity_type_string = "1.0"
    elif continuity_type == "C2":
        continuity_type_string = "2.0"
    else:
        raise ValueError("Continuity type must be 'C0', 'C1', or 'C2'.")

    for i, xsec in enumerate(fuselage.xsecs):
        script += f"""\
// ASB Section {i}, VSP Section {i}
SetParmVal( fid, "XLocPercent", "XSec_{i}", {xsec.xyz_c[0] / length} );
SetParmVal( fid, "YLocPercent", "XSec_{i}", {xsec.xyz_c[1] / length} );
SetParmVal( fid, "ZLocPercent", "XSec_{i}", {xsec.xyz_c[2] / length} );
SetParmVal( fid, "SectTess_U", "XSec_{i}", 10 );
SetParmVal( fid, "AllSym", "XSec_{i}", 1.0 );
SetParmVal( fid, "TopLAngleSet", "XSec_{i}", 0.0 );
SetParmVal( fid, "TopLAngle", "XSec_{i}", 0.0 );
SetParmVal( fid, "TopLStrengthSet", "XSec_{i}", 0.0 );
SetParmVal( fid, "TopLStrength", "XSec_{i}", 0.5 );
SetParmVal( fid, "ContinuityTop", "XSec_{i}", {continuity_type_string} );
Update();

"""

    script += """\
//==== Set Shapes ====//
"""

    for i in range(len(fuselage.xsecs)):
        xsec = fuselage.xsecs[i]
        script += f"""\
{{
    string xsec_surf = GetXSecSurf( fid, {i} );
    xsec_surf = GetXSecSurf( fid, {i} );
    string xsec = GetXSec( xsec_surf, {i} );
    SetParmVal( fid, "Super_Width", "XSecCurve_{i}", {xsec.width} );
    SetParmVal( fid, "Super_Height", "XSecCurve_{i}", {xsec.height} );
    SetParmVal( fid, "Super_M", "XSecCurve_{i}", {xsec.shape} );
    SetParmVal( fid, "Super_N", "XSecCurve_{i}", {xsec.shape} );
    Update();
}}
"""

    script += """\
SetParmVal( fid, "Tess_W", "Shape", 100 );
Update();
    """

    if include_main:
        script = _utilities.wrap_script(script)

    return script


if __name__ == '__main__':
    import aerosandbox as asb
    import aerosandbox.numpy as np

    af = asb.Airfoil("ls013")
    x = np.linspace(0, 1, 10)

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
    print(generate_fuselage(fuse))

    with open("test.vspscript", "w") as f:
        f.write(generate_fuselage(fuse))
