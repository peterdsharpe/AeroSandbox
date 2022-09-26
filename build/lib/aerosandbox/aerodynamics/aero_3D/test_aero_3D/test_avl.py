import aerosandbox as asb
import pytest


def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""

    # from whichcraft import which
    from shutil import which

    return which(name) is not None


avl_present = is_tool('avl')


def test_conventional():
    if not avl_present:
        return
    from aerosandbox.aerodynamics.aero_3D.test_aero_3D.geometries.conventional import airplane
    analysis = asb.AVL(
        airplane=airplane,
        op_point=asb.OperatingPoint(alpha=10),
    )
    return analysis.run()


def test_vanilla():
    if not avl_present:
        return
    from aerosandbox.aerodynamics.aero_3D.test_aero_3D.geometries.vanilla import airplane
    analysis = asb.AVL(
        airplane=airplane,
        op_point=asb.OperatingPoint(alpha=10),
    )
    return analysis.run()


def test_flat_plate():
    if not avl_present:
        return
    from aerosandbox.aerodynamics.aero_3D.test_aero_3D.geometries.flat_plate import airplane
    analysis = asb.AVL(
        airplane=airplane,
        op_point=asb.OperatingPoint(alpha=10),
    )
    return analysis.run()


def test_flat_plate_mirrored():
    if not avl_present:
        return
    from aerosandbox.aerodynamics.aero_3D.test_aero_3D.geometries.flat_plate_mirrored import airplane
    analysis = asb.AVL(
        airplane=airplane,
        op_point=asb.OperatingPoint(alpha=10),
    )
    return analysis.run()


if __name__ == '__main__':
    # test_conventional()
    # test_vanilla()

    print(test_flat_plate()['CL'])
    # pytest.main()
