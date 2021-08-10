import aerosandbox as asb
import pytest

def test_conventional():
    from aerosandbox.aerodynamics.aero_3D.test_aero_3D.geometries.conventional import airplane
    analysis = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=asb.OperatingPoint(alpha=10),
    )
    return analysis.run()

if __name__ == '__main__':
    test_conventional()
    # pytest.main()