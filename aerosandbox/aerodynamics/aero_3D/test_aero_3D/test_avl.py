import aerosandbox as asb
import aerosandbox.numpy as np


def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""

    # from whichcraft import which
    from shutil import which

    return which(name) is not None


avl_present = is_tool("avl")


def test_conventional():
    if not avl_present:
        return
    from aerosandbox.aerodynamics.aero_3D.test_aero_3D.geometries.conventional import (
        airplane,
    )

    analysis = asb.AVL(
        airplane=airplane,
        op_point=asb.OperatingPoint(alpha=10),
    )
    return analysis.run()


def test_vanilla():
    if not avl_present:
        return
    from aerosandbox.aerodynamics.aero_3D.test_aero_3D.geometries.vanilla import (
        airplane,
    )

    analysis = asb.AVL(
        airplane=airplane,
        op_point=asb.OperatingPoint(alpha=10),
    )
    return analysis.run()


def test_flat_plate():
    if not avl_present:
        return
    from aerosandbox.aerodynamics.aero_3D.test_aero_3D.geometries.flat_plate import (
        airplane,
    )

    analysis = asb.AVL(
        airplane=airplane,
        op_point=asb.OperatingPoint(alpha=10),
    )
    return analysis.run()


def test_flat_plate_mirrored():
    if not avl_present:
        return
    from aerosandbox.aerodynamics.aero_3D.test_aero_3D.geometries.flat_plate_mirrored import (
        airplane,
    )

    analysis = asb.AVL(
        airplane=airplane,
        op_point=asb.OperatingPoint(alpha=10),
    )
    return analysis.run()


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
    # test_conventional()
    # test_vanilla()

    print(test_flat_plate()["CL"])
    # pytest.main()
