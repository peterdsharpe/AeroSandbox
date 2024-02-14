import aerosandbox.numpy as np
from aerosandbox.geometry.wing import Wing, WingXSec
from aerosandbox.geometry.airfoil.airfoil import Airfoil
from textwrap import indent, dedent
from aerosandbox.geometry.openvsp_io.asb_to_openvsp import _utilities


def generate_wing(
        wing: Wing,
        include_main=True
) -> str:
    """
    Generates a VSPScript file for a Wing object.

    Args:
        wing: The Wing object to generate a VSPScript file for.

    Returns: A VSPScript file as a string.

    """
    script = ""

    script += f"""\
//==== Add Wing "{wing.name}" ====//
string wid = AddGeom("WING");
ChangeXSecShape( GetXSecSurf( wid, 0 ), 0, XS_FILE_AIRFOIL );
ChangeXSecShape( GetXSecSurf( wid, 1 ), 1, XS_FILE_AIRFOIL );

"""

    script += """//==== Generate Blank Wing Sections ====//\n"""
    if len(wing.xsecs) < 2:
        raise ValueError("Wings must have at least 2 cross sections.")
    else:
        for i in range(len(wing.xsecs) - 1):
            script += f"""\
InsertXSec( wid, 1, XS_FILE_AIRFOIL ); // WingXSecs {i} to {i + 1}
"""

    script += f"""\
//===== Cut The Original Section =====//
CutXSec( wid, 1 );

//==== Set Overall Wing Options and First Section ====//
SetParmVal( wid, "X_Rel_Location", "XForm", {wing.xsecs[0].xyz_le[0]} );
SetParmVal( wid, "Y_Rel_Location", "XForm", {wing.xsecs[0].xyz_le[1]} );
SetParmVal( wid, "Z_Rel_Location", "XForm", {wing.xsecs[0].xyz_le[2]} );
SetParmVal( wid, "Twist", "XSec_0", {wing.xsecs[0].twist} );
SetParmVal( wid, "Twist_Location", "XSec_0", 0.0 );
SetParmVal( wid, "Sym_Ancestor", "Sym", 0.0 ); // Global Origin
SetParmVal( wid, "Sym_Planar_Flag", "Sym", {"2.0" if wing.symmetric else "0.0"} ); // 0 = None, 2 = XZ sym.
SetParmVal( wid, "RotateAirfoilMatchDideralFlag", "WingGeom", 1.0 ); // sic, typo from OpenVSP
Update();

"""

    script += """//==== Set Wing Section Options ====//\n"""
    xyz_le = np.stack([xsec.xyz_le for xsec in wing.xsecs], axis=0)
    dxyz_le = np.diff(xyz_le, axis=0)
    dx_le = dxyz_le[:, 0]
    dy_le = dxyz_le[:, 1]
    dz_le = dxyz_le[:, 2]
    dyz_le = np.sqrt(dy_le ** 2 + dz_le ** 2)
    dihedrals = np.arctan2d(dz_le, dy_le)
    sweep_le = np.arctan2d(dx_le, dyz_le)

    for i, (xsec_a, xsec_b) in enumerate(zip(wing.xsecs[:-1], wing.xsecs[1:])):
        script += f"""\
// ASB Section {i}, VSP Section {i + 1} (ASB XSecs {i} to {i + 1})
SetParmVal( wid, "Root_Chord", "XSec_{i + 1}", {xsec_a.chord} );
SetParmVal( wid, "Tip_Chord", "XSec_{i + 1}", {xsec_b.chord} );
SetParmVal( wid, "Span", "XSec_{i + 1}", {dyz_le[i]});
SetParmVal( wid, "Sweep", "XSec_{i + 1}", {sweep_le[i]} );
SetParmVal( wid, "Sweep_Location", "XSec_{i + 1}", 0.0 );
SetParmVal( wid, "Twist", "XSec_{i + 1}", {xsec_b.twist} );
SetParmVal( wid, "Twist_Location", "XSec_{i + 1}", 0.0 );
SetParmVal( wid, "Dihedral", "XSec_{i + 1}", {dihedrals[i]} );
SetParmVal( wid, "SectTess_U", "XSec_{i + 1}", 10 );
Update();

"""

    script += """\
//==== Set Airfoils ====//
"""

    for i in range(len(wing.xsecs)):
        xsec = wing.xsecs[i]

        upper = xsec.airfoil.upper_coordinates()[::-1]  # VSP wants front to back order
        lower = xsec.airfoil.lower_coordinates()  # Front to back order

        up_pnt_vecs = ", ".join([
            f"vec3d({p[0]:.8g}, {p[1]:.8g}, 0.0)" for p in upper
        ])
        lo_pnt_vecs = ", ".join([
            f"vec3d({p[0]:.8g}, {p[1]:.8g}, 0.0)" for p in lower
        ])

        script += f"""\
{{
    array<vec3d> up_pnt_vec = {{
        {up_pnt_vecs}
    }};
    array<vec3d> lo_pnt_vec = {{
        {lo_pnt_vecs}
    }};
    string xsec_surf = GetXSecSurf( wid, {i} );
    xsec_surf = GetXSecSurf( wid, {i} );
    string xsec = GetXSec( xsec_surf, {i} );
    SetParmVal( wid, "ThickChord", "XSecCurve_{i}", 0.5 ); // This is a hack that triggers VSP to update the curve
    SetParmVal( wid, "ThickChord", "XSecCurve_{i}", 1.0 );
    SetAirfoilPnts( xsec, up_pnt_vec, lo_pnt_vec );
    Update();
}}
"""

    script += """\
SetParmVal( wid, "Tess_W", "Shape", 100 );
Update();
"""

    if include_main:
        script = _utilities.wrap_script(script)

    return script


if __name__ == '__main__':
    wing = Wing(
        name="Main Wing",
        xsecs=[
            WingXSec(
                xyz_le=[1, 0, 0],
                chord=1.1,
                twist=-20,
                airfoil=Airfoil(name="dae11")
            ),
            WingXSec(
                xyz_le=[1, 2, 0],
                chord=0.9,
                twist=40,
                airfoil=Airfoil(name="NACA4412")
            ),
            WingXSec(
                xyz_le=[2, 5, 0],
                chord=0.5,
                twist=0,
                airfoil=Airfoil(name="NACA3412")
            ),
            WingXSec(
                xyz_le=[2, 10, 5],
                chord=0.25,
                twist=0,
                airfoil=Airfoil(name="NACA2412")
            )
        ]
    )
    print(generate_wing(wing))

    with open("test.vspscript", "w") as f:
        f.write(generate_wing(wing))
