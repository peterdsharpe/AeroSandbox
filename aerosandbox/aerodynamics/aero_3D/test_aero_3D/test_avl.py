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
