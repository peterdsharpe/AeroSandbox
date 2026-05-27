import aerosandbox as asb
import aerosandbox.numpy as np
import numpy as onp
import pytest


def _control_surface_airplane(control_surfaces):
    """A simple symmetric rectangular wing, with the given control surfaces on its inboard xsec."""
    airfoil = asb.Airfoil("naca0008")
    return asb.Airplane(
        xyz_ref=[0.25, 0, 0],
        wings=[
            asb.Wing(
                name="Wing",
                symmetric=True,
                xsecs=[
                    asb.WingXSec(
                        xyz_le=[0, 0, 0],
                        chord=1,
                        airfoil=airfoil,
                        control_surfaces=control_surfaces,
                    ),
                    asb.WingXSec(
                        xyz_le=[0, 5, 0],
                        chord=1,
                        airfoil=airfoil,
                    ),
                ],
            )
        ],
    )


def _run_control_surface_case(control_surfaces, alpha=0.0):
    return asb.VortexLatticeMethod(
        airplane=_control_surface_airplane(control_surfaces),
        op_point=asb.OperatingPoint(velocity=10, alpha=alpha),
        spanwise_resolution=8,
        chordwise_resolution=10,
    ).run()


def test_control_surface_zero_deflection_matches_baseline():
    """A control surface at zero deflection must not change the result vs. having no control surface."""
    baseline = _run_control_surface_case([], alpha=5)
    zero_defl = _run_control_surface_case(
        [asb.ControlSurface(name="flap", hinge_point=0.75, deflection=0)], alpha=5
    )
    for key in ["CL", "CD", "Cl", "Cm", "Cn"]:
        assert zero_defl[key] == pytest.approx(baseline[key], abs=1e-12)


def test_control_surface_symmetric_flap_increases_lift():
    """A downward-deflected symmetric flap should monotonically increase CL (with ~zero roll)."""
    CLs = [
        _run_control_surface_case(
            [
                asb.ControlSurface(
                    name="flap", symmetric=True, hinge_point=0.75, deflection=d
                )
            ],
            alpha=0,
        )
        for d in [0, 5, 10, 20]
    ]
    # CL increases monotonically with downward deflection, starting from ~0 at alpha=0, deflection=0.
    assert CLs[0]["CL"] == pytest.approx(0, abs=1e-9)
    assert CLs[0]["CL"] < CLs[1]["CL"] < CLs[2]["CL"] < CLs[3]["CL"]
    # A symmetric flap produces no net rolling moment.
    for aero in CLs:
        assert aero["Cl"] == pytest.approx(0, abs=1e-9)


def test_control_surface_aileron_produces_roll():
    """An antisymmetric aileron should produce a rolling moment with ~zero net lift."""
    aero = _run_control_surface_case(
        [
            asb.ControlSurface(
                name="aileron", symmetric=False, hinge_point=0.75, deflection=10
            )
        ],
        alpha=0,
    )
    assert abs(aero["Cl"]) > 0.01  # Nonzero rolling moment.
    assert aero["CL"] == pytest.approx(
        0, abs=1e-9
    )  # Net lift cancels between the two sides.


def test_control_surface_deflection_is_differentiable():
    """The deflection angle must be usable as an optimization variable (the analysis stays differentiable)."""
    opti = asb.Opti()
    deflection = opti.variable(init_guess=5.0)
    airfoil = asb.Airfoil("naca0008")
    airplane = asb.Airplane(
        xyz_ref=[0.25, 0, 0],
        wings=[
            asb.Wing(
                name="Wing",
                symmetric=True,
                xsecs=[
                    asb.WingXSec(
                        xyz_le=[0, 0, 0],
                        chord=1,
                        airfoil=airfoil,
                        control_surfaces=[
                            asb.ControlSurface(
                                name="flap", hinge_point=0.7, deflection=deflection
                            )
                        ],
                    ),
                    asb.WingXSec(xyz_le=[0, 5, 0], chord=1, airfoil=airfoil),
                ],
            )
        ],
    )
    aero = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=asb.OperatingPoint(velocity=10, alpha=0),
        spanwise_resolution=6,
        chordwise_resolution=8,
    ).run()
    opti.subject_to(aero["CL"] == 0.5)
    sol = opti.solve(verbose=False)
    assert np.isfinite(sol(deflection))
    assert sol(aero["CL"]) == pytest.approx(0.5, abs=1e-6)


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

    ### Check lift against a documented analytic reference:
    # For a thin flat plate of aspect ratio 10, the Helmbold finite-wing correction to thin-airfoil
    # theory (see e.g. Anderson, "Fundamentals of Aerodynamics", low-AR wing lift slope estimates)
    # gives CL_alpha ~= 2 * pi / (1 + 2 / AR), so CL(10 deg) ~= 0.914. The VLM solution is expected
    # to be close to (within several percent of) this estimate.
    aspect_ratio = 10
    CL_helmbold = 2 * onp.pi / (1 + 2 / aspect_ratio) * onp.radians(10)
    assert aero["CL"] == pytest.approx(CL_helmbold, rel=0.1)

    # Pin the current computed value (0.87012, which is 4.8% below the Helmbold estimate) tightly,
    # so that unintended changes to the VLM solver's numerics are caught:
    assert aero["CL"] == pytest.approx(0.870116, rel=1e-3)


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
