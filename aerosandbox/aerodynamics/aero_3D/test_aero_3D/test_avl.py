import aerosandbox as asb
import aerosandbox.numpy as np
import pytest


def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""

    # from whichcraft import which
    from shutil import which

    return which(name) is not None


avl_present = is_tool("avl")

requires_avl = pytest.mark.skipif(
    not avl_present,
    reason="The AVL executable is not on PATH; try installing AVL from https://web.mit.edu/drela/Public/web/avl/",
)


def sanity_check_results(res: dict) -> None:
    """
    Basic sanity checks on AVL results for a lifting airplane at alpha = 10 deg.
    """
    assert res["alpha"] == pytest.approx(10, abs=1e-3)
    assert res["CL"] > 0
    assert res["CD"] > 0
    for key in ["CL", "CD", "Cm", "L", "D"]:
        assert np.isfinite(res[key])


@requires_avl
def test_conventional():
    from aerosandbox.aerodynamics.aero_3D.test_aero_3D.geometries.conventional import (
        airplane,
    )

    analysis = asb.AVL(
        airplane=airplane,
        op_point=asb.OperatingPoint(alpha=10),
    )
    sanity_check_results(analysis.run())


@requires_avl
def test_vanilla():
    from aerosandbox.aerodynamics.aero_3D.test_aero_3D.geometries.vanilla import (
        airplane,
    )

    analysis = asb.AVL(
        airplane=airplane,
        op_point=asb.OperatingPoint(alpha=10),
    )
    sanity_check_results(analysis.run())


@requires_avl
def test_flat_plate():
    from aerosandbox.aerodynamics.aero_3D.test_aero_3D.geometries.flat_plate import (
        airplane,
    )

    analysis = asb.AVL(
        airplane=airplane,
        op_point=asb.OperatingPoint(alpha=10),
    )
    sanity_check_results(analysis.run())


@requires_avl
def test_flat_plate_mirrored():
    from aerosandbox.aerodynamics.aero_3D.test_aero_3D.geometries.flat_plate_mirrored import (
        airplane,
    )

    analysis = asb.AVL(
        airplane=airplane,
        op_point=asb.OperatingPoint(alpha=10),
    )
    sanity_check_results(analysis.run())


def make_flapped_wing_airplane(
    aileron_deflection: float = 0.0,
    elevator_deflection: float = 0.0,
) -> asb.Airplane:
    """
    A simple two-surface airplane with an antisymmetric aileron on the wing and a
    symmetric elevator on the horizontal tail.
    """
    wing = asb.Wing(
        name="Main Wing",
        symmetric=True,
        xsecs=[
            asb.WingXSec(
                xyz_le=[0, 0, 0],
                chord=1,
                airfoil=asb.Airfoil("naca0012"),
                control_surfaces=[
                    asb.ControlSurface(
                        name="Aileron",
                        symmetric=False,
                        hinge_point=0.75,
                        deflection=aileron_deflection,
                    )
                ],
            ),
            asb.WingXSec(xyz_le=[0, 4, 0], chord=1, airfoil=asb.Airfoil("naca0012")),
        ],
    )
    htail = asb.Wing(
        name="H Tail",
        symmetric=True,
        xsecs=[
            asb.WingXSec(
                xyz_le=[0, 0, 0],
                chord=0.5,
                airfoil=asb.Airfoil("naca0012"),
                control_surfaces=[
                    asb.ControlSurface(
                        name="Elevator",
                        symmetric=True,
                        hinge_point=0.75,
                        deflection=elevator_deflection,
                    )
                ],
            ),
            asb.WingXSec(
                xyz_le=[0, 1.5, 0], chord=0.5, airfoil=asb.Airfoil("naca0012")
            ),
        ],
    ).translate([4, 0, 0])
    return asb.Airplane(
        name="TwoSurf",
        wings=[wing, htail],
        s_ref=8,
        c_ref=1,
        b_ref=8,
    )


def test_control_surface_deflections_written_to_avl_file():
    """
    Regression test for https://github.com/peterdsharpe/AeroSandbox/issues/134:

    Control surface deflections are passed to AVL by attaching every control surface
    to a single AVL control variable ("all_deflections"), with each surface's
    user-specified deflection written as that surface's CONTROL-card gain
    [deg deflection / unit control variable]. The run keystrokes then set that
    control variable to 1 ('d1 d1 1'), deflecting each surface by its own amount.

    Commit 8704f5df accidentally changed the CONTROL cards to per-surface names with
    gain = 1 (without updating the keystrokes), which made AVL silently ignore all
    user-specified deflections and instead deflect the first control surface by
    exactly 1 degree.
    """
    analysis = asb.AVL(
        airplane=make_flapped_wing_airplane(
            aileron_deflection=5,
            elevator_deflection=-10.5,
        ),
        op_point=asb.OperatingPoint(velocity=10, alpha=3),
    )

    avl_string = analysis.write_avl()  # String mode; no files written

    control_lines = [
        line
        for line in avl_string.split("\n")
        if line.startswith("all_deflections")
    ]

    # Each control surface writes its CONTROL card to both bounding sections.
    # Aileron: gain = 5, antisymmetric (SgnDup = -1)
    assert control_lines.count("all_deflections 5 0.75 0 0 0 -1") == 2
    # Elevator: gain = -10.5, symmetric (SgnDup = +1)
    assert control_lines.count("all_deflections -10.5 0.75 0 0 0 1") == 2
    assert len(control_lines) == 4

    # The keystrokes must set the shared control variable to exactly 1, so that
    # each surface deflects by (gain * 1) = its own user-specified deflection.
    assert "d1 d1 1" in analysis._default_keystroke_file_contents()


@requires_avl
def test_control_surface_deflection_changes_avl_results():
    """
    Regression test (behavioral) for
    https://github.com/peterdsharpe/AeroSandbox/issues/134: deflecting a symmetric
    trailing-edge elevator downwards must increase lift and pitch the airplane
    nose-down, relative to the undeflected airplane at the same operating point.
    """
    op_point = asb.OperatingPoint(velocity=10, alpha=3)

    res_undeflected = asb.AVL(
        airplane=make_flapped_wing_airplane(elevator_deflection=0),
        op_point=op_point,
    ).run()
    res_deflected = asb.AVL(
        airplane=make_flapped_wing_airplane(elevator_deflection=10),
        op_point=op_point,
    ).run()

    assert res_deflected["CL"] > res_undeflected["CL"] + 0.01
    assert res_deflected["Cm"] < res_undeflected["Cm"] - 0.01


def test_write_avl_string_mode(tmp_path):
    """
    Regression test: write_avl(filepath=None) used to raise TypeError (from
    Path(None)) despite the docstring promising a string return, and wrote
    airfoil sidecar files to literal 'None.af0' paths in the current directory.
    """
    import os

    from aerosandbox.aerodynamics.aero_3D.test_aero_3D.geometries.vanilla import (
        airplane,
    )

    analysis = asb.AVL(
        airplane=airplane,
        op_point=asb.OperatingPoint(alpha=10),
    )

    original_cwd = os.getcwd()
    os.chdir(tmp_path)  # So any (undesired) sidecar file writes are detectable
    try:
        avl_string = analysis.write_avl()  # filepath=None -> string mode
        assert isinstance(avl_string, str)
        assert "SURFACE" in avl_string
        assert airplane.name in avl_string
        assert os.listdir(tmp_path) == []  # String mode must not write any files
    finally:
        os.chdir(original_cwd)

    # File mode should write the same contents, and also return them
    filepath = tmp_path / "airplane.avl"
    returned_string = analysis.write_avl(filepath)
    assert filepath.is_file()
    assert filepath.read_text() == returned_string

    # write_avl_bfile has the same string-mode contract
    bfile_string = asb.AVL.write_avl_bfile(fuselage=airplane.fuselages[0])
    assert isinstance(bfile_string, str)
    assert airplane.fuselages[0].name in bfile_string


def test_parse_unformatted_data_output_at_end_of_string():
    """
    Regression test: values (or trailing blanks) that end exactly at the end of
    the string used to raise IndexError, since the scan loops indexed `s[i]`
    before checking bounds.
    """
    # Value ends exactly at end-of-string (no trailing newline)
    assert asb.AVL.parse_unformatted_data_output("CLtot =   1.01454") == {
        "CLtot": 1.01454
    }

    # Multiline, with the last value at end-of-string
    assert asb.AVL.parse_unformatted_data_output("CLtot = 1.0\nCDtot = 0.5") == {
        "CLtot": 1.0,
        "CDtot": 0.5,
    }

    # Identifier followed only by blanks to end-of-string
    result = asb.AVL.parse_unformatted_data_output("CLtot =    ")
    assert list(result.keys()) == ["CLtot"]
    assert np.isnan(result["CLtot"])

    # A realistic AVL output block should parse identically with or without a
    # trailing newline.
    s = (
        "  Alpha =   0.43348     pb/2V =  -0.00000\n"
        "  CLtot =   1.01454\n"
        "  CDvis =   0.00000     CDind = 0.0291513\n"
    )
    parsed_with_newline = asb.AVL.parse_unformatted_data_output(s)
    parsed_without_newline = asb.AVL.parse_unformatted_data_output(s.rstrip("\n"))
    assert parsed_with_newline == parsed_without_newline
    assert parsed_with_newline["CLtot"] == 1.01454


if __name__ == "__main__":
    pytest.main()
