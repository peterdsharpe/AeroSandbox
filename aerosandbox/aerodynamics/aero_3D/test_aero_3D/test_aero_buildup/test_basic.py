import aerosandbox as asb
from aerosandbox.aerodynamics.aero_3D.test_aero_3D.geometries.conventional import (
    airplane,
)


def test_aero_buildup():
    analysis = asb.AeroBuildup(
        airplane=airplane,
        op_point=asb.OperatingPoint(),
    )
    aero = analysis.run()
    assert aero is not None  ### Verify analysis produces results


if __name__ == "__main__":
    test_aero_buildup()
    # pytest.main()
