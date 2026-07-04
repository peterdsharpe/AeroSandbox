import aerosandbox as asb
import pytest
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


def test_symmetric_wing_lateral_quantities_are_zero():
    """
    A symmetric wing in symmetric flight (beta = p = r = 0) must produce zero side force,
    rolling moment, and yawing moment. This exercises the mirrored-section branch of
    AeroBuildup.wing_aerodynamics (mirror_across_XZ), which must not corrupt the shared
    per-section aerodynamic-center data.
    """
    wing = asb.Wing(
        symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[0, 0, 0], chord=1.0, airfoil=asb.Airfoil("naca4412")),
            asb.WingXSec(
                xyz_le=[0.2, 5, 0.5], chord=0.6, airfoil=asb.Airfoil("naca4412")
            ),
        ],
    )
    test_airplane = asb.Airplane(wings=[wing], s_ref=8, c_ref=0.8, b_ref=10)
    aero = asb.AeroBuildup(
        airplane=test_airplane,
        op_point=asb.OperatingPoint(velocity=25, alpha=5),
        xyz_ref=[0.3, 0, 0],
    ).run()

    assert aero["F_b"][1] == pytest.approx(0, abs=1e-8)  # Side force
    assert aero["l_b"] == pytest.approx(0, abs=1e-8)  # Rolling moment
    assert aero["n_b"] == pytest.approx(0, abs=1e-8)  # Yawing moment
    assert aero["L"] > 0  # Sanity check: the wing is lifting


if __name__ == "__main__":
    test_aero_buildup()
    test_symmetric_wing_lateral_quantities_are_zero()
    # pytest.main()
