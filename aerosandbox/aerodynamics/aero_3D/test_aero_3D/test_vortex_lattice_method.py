import aerosandbox as asb
import pytest


def test_conventional():
    from aerosandbox.aerodynamics.aero_3D.test_aero_3D.geometries.conventional import airplane
    analysis = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=asb.OperatingPoint(alpha=10),
    )
    return analysis.run()


def test_vanilla():
    from aerosandbox.aerodynamics.aero_3D.test_aero_3D.geometries.vanilla import airplane
    analysis = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=asb.OperatingPoint(alpha=10),
    )
    return analysis.run()


def test_flat_plate():
    from aerosandbox.aerodynamics.aero_3D.test_aero_3D.geometries.flat_plate import airplane
    analysis = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=asb.OperatingPoint(alpha=10),
    )
    return analysis.run()


def test_flat_plate_mirrored():
    from aerosandbox.aerodynamics.aero_3D.test_aero_3D.geometries.flat_plate_mirrored import airplane
    analysis = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=asb.OperatingPoint(alpha=10),
        spanwise_resolution=1,
        chordwise_resolution=3,
    )
    return analysis.run()


if __name__ == '__main__':
    # test_conventional()
    # test_vanilla()
    # test_flat_plate()['CL']
    # test_flat_plate_mirrored()
    # pytest.main()
    from aerosandbox.aerodynamics.aero_3D.test_aero_3D.geometries.conventional import airplane

    analysis = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=asb.OperatingPoint(alpha=10),
    )
    aero = analysis.run()
    analysis.draw()
