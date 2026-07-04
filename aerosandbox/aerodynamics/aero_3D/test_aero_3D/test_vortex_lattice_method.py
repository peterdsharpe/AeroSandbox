import aerosandbox as asb
import numpy as onp
import pytest


def test_conventional():
    from aerosandbox.aerodynamics.aero_3D.test_aero_3D.geometries.conventional import (
        airplane,
    )

    analysis = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=asb.OperatingPoint(alpha=10),
    )
    aero = analysis.run()
    assert aero is not None  ### Verify analysis produces results


def test_vanilla():
    from aerosandbox.aerodynamics.aero_3D.test_aero_3D.geometries.vanilla import (
        airplane,
    )

    analysis = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=asb.OperatingPoint(alpha=10),
    )
    aero = analysis.run()
    assert aero is not None  ### Verify analysis produces results


def test_flat_plate():
    from aerosandbox.aerodynamics.aero_3D.test_aero_3D.geometries.flat_plate import (
        airplane,
    )

    analysis = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=asb.OperatingPoint(alpha=10),
    )
    aero = analysis.run()
    assert aero is not None  ### Verify analysis produces results


def test_flat_plate_mirrored():
    from aerosandbox.aerodynamics.aero_3D.test_aero_3D.geometries.flat_plate_mirrored import (
        airplane,
    )

    analysis = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=asb.OperatingPoint(alpha=10),
        spanwise_resolution=1,
        chordwise_resolution=3,
    )
    aero = analysis.run()
    assert aero is not None  ### Verify analysis produces results


def test_run_with_stability_derivatives_zero_division_guard():
    """
    If a (degenerate) configuration produces exactly-zero CLa or CYb, the neutral-point computations in
    run_with_stability_derivatives should return NaN, rather than raising ZeroDivisionError or returning
    +/-inf. (This matches the behavior of the AeroBuildup and LiftingLine implementations.)
    """

    class MockZeroLiftSlopeVLM(asb.VortexLatticeMethod):
        """
        A mock analysis with exactly-zero lift-slope and sideforce-slope (CLa == 0, CYb == 0), but
        nonzero moment derivatives (Cma != 0, Cnb != 0) -- so an unguarded division would produce
        +/-inf (or ZeroDivisionError) in the neutral-point computations.
        """

        def run(self):
            return {
                "CL": 0.0,
                "CD": 0.0,
                "CY": 0.0,
                "Cl": 0.0,
                "Cm": -0.01 * self.op_point.alpha,
                "Cn": 0.01 * self.op_point.beta,
            }

    airplane = asb.Airplane(
        wings=[
            asb.Wing(
                symmetric=True,
                xsecs=[
                    asb.WingXSec(
                        xyz_le=[0, 0, 0], chord=1, airfoil=asb.Airfoil("naca0012")
                    ),
                    asb.WingXSec(
                        xyz_le=[0, 5, 0], chord=1, airfoil=asb.Airfoil("naca0012")
                    ),
                ],
            )
        ],
        s_ref=10.0,
        c_ref=1.0,
        b_ref=10.0,
    )

    vlm = MockZeroLiftSlopeVLM(
        airplane=airplane,
        op_point=asb.OperatingPoint(velocity=10),
    )
    aero = vlm.run_with_stability_derivatives()  # Should not raise

    assert onp.isnan(aero["x_np"])
    assert onp.isnan(aero["x_np_lateral"])


if __name__ == "__main__":
    # test_conventional()
    # test_vanilla()
    # test_flat_plate()['CL']
    # test_flat_plate_mirrored()
    # pytest.main()
    from aerosandbox.aerodynamics.aero_3D.test_aero_3D.geometries.conventional import (
        airplane,
    )

    analysis = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=asb.OperatingPoint(alpha=10),
    )
    aero = analysis.run()
    analysis.draw()
