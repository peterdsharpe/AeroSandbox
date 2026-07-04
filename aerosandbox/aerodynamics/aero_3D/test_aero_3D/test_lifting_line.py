import aerosandbox as asb
import aerosandbox.numpy as np
import pytest

### Canonical test case: a rectangular, planar, constant-chord, symmetric wing
### with a symmetric airfoil, at a small angle of attack.
AR = 10  # Aspect ratio
b = 10.0  # Span [m]
c = 1.0  # Chord [m]
alpha_deg = 5.0  # Angle of attack [deg]

airfoil = asb.Airfoil("naca0012")

### Classical (Prandtl) lifting-line expectations for a rectangular wing.
### See, e.g., Anderson, "Fundamentals of Aerodynamics", Ch. 5: for a rectangular
### wing of AR ~= 10, the lift-slope correction factor tau ~= 0.05 and the
### induced-drag factor delta ~= 0.049 (span efficiency e ~= 0.95).
_a0 = 2 * np.pi  # Thin-airfoil-theory 2D lift-curve slope [1/rad]
CL_expected = (_a0 / (1 + _a0 / (np.pi * AR) * (1 + 0.05))) * np.radians(alpha_deg)
CDi_expected = CL_expected**2 / (np.pi * AR) * (1 + 0.049)


def make_rectangular_wing_airplane() -> asb.Airplane:
    return asb.Airplane(
        wings=[
            asb.Wing(
                symmetric=True,
                xsecs=[
                    asb.WingXSec(xyz_le=[0, 0, 0], chord=c, airfoil=airfoil),
                    asb.WingXSec(xyz_le=[0, b / 2, 0], chord=c, airfoil=airfoil),
                ],
            )
        ],
        s_ref=b * c,
        b_ref=b,
        c_ref=c,
        xyz_ref=[0.25 * c, 0, 0],
    )


def make_op_point() -> asb.OperatingPoint:
    return asb.OperatingPoint(
        velocity=25,  # [m/s]; gives Re ~ 1.7e6, deep in the subsonic regime.
        alpha=alpha_deg,
    )


def section_profile_drag_estimate() -> float:
    """
    Estimates the 2D (profile) drag coefficient of the wing section, evaluated at the
    effective angle of attack (geometric alpha minus the analytic induced alpha), using
    the same 2D aerodynamics model (NeuralFoil) that the lifting-line solvers use.
    """
    op_point = make_op_point()
    alpha_induced_deg = np.degrees(CL_expected / (np.pi * AR))
    section_aero = airfoil.get_aero_from_neuralfoil(
        alpha=alpha_deg - alpha_induced_deg,
        Re=op_point.velocity * c / op_point.atmosphere.kinematic_viscosity(),
        mach=op_point.mach(),
        model_size="medium",
    )
    return float(np.mean(section_aero["CD"]))


def test_lifting_line_rectangular_wing():
    aero = asb.LiftingLine(
        airplane=make_rectangular_wing_airplane(),
        op_point=make_op_point(),
    ).run()

    ### Lift should match classical lifting-line theory closely.
    assert aero["CL"] == pytest.approx(CL_expected, rel=0.05)

    ### LiftingLine reports only total drag (induced + profile), so back out the
    ### induced component using a 2D profile-drag estimate for the same section.
    CDi_inferred = aero["CD"] - section_profile_drag_estimate()
    assert CDi_inferred == pytest.approx(CDi_expected, rel=0.3)


def test_nonlinear_lifting_line_rectangular_wing():
    aero = asb.NonlinearLiftingLine(
        airplane=make_rectangular_wing_airplane(),
        op_point=make_op_point(),
    ).run()

    ### Lift should match classical lifting-line theory closely.
    assert aero["CL"] == pytest.approx(CL_expected, rel=0.05)

    ### NonlinearLiftingLine reports the induced-drag component directly.
    assert aero["CDi"] == pytest.approx(CDi_expected, rel=0.3)


def make_aileron_airplane(deflection: float) -> asb.Airplane:
    return asb.Airplane(
        wings=[
            asb.Wing(
                symmetric=True,
                xsecs=[
                    asb.WingXSec(
                        xyz_le=[0, 0, 0],
                        chord=c,
                        airfoil=airfoil,
                        control_surfaces=[
                            asb.ControlSurface(
                                name="Aileron",
                                symmetric=False,  # Antisymmetric deflection (mirrored surface deflects opposite)
                                deflection=deflection,
                            )
                        ],
                    ),
                    asb.WingXSec(xyz_le=[0, b / 2, 0], chord=c, airfoil=airfoil),
                ],
            )
        ],
        s_ref=b * c,
        b_ref=b,
        c_ref=c,
        xyz_ref=[0.25 * c, 0, 0],
    )


@pytest.mark.parametrize(
    "AnalysisClass",
    [asb.LiftingLine, asb.NonlinearLiftingLine],
)
def test_symmetric_wing_with_asymmetric_control_surface(AnalysisClass):
    """
    Runs each solver on a symmetric wing with an asymmetrically-deflected control surface.

    This is a regression test for a NameError crash: both solvers define a nested
    `mirror_control_surface()` helper (annotated with `ControlSurface`) when meshing any
    symmetric wing, and that annotation must be resolvable at def-time regardless of
    whether `ControlSurface` is imported at runtime or only under `typing.TYPE_CHECKING`.

    It also checks the physics of the control-surface mirroring itself: an
    antisymmetric ("aileron-style") deflection should produce a rolling moment.
    """
    op_point = asb.OperatingPoint(velocity=25, alpha=3)

    aero_undeflected = AnalysisClass(
        airplane=make_aileron_airplane(deflection=0),
        op_point=op_point,
    ).run()
    assert aero_undeflected["Cl"] == pytest.approx(0, abs=1e-6)

    aero_deflected = AnalysisClass(
        airplane=make_aileron_airplane(deflection=10),
        op_point=op_point,
    ).run()

    ### A positive (trailing-edge-down, on the right side) aileron deflection increases
    ### lift on the right wing, rolling the aircraft to the left (negative Cl).
    assert aero_deflected["Cl"] < -0.05


@pytest.mark.parametrize(
    "AnalysisClass",
    [asb.LiftingLine, asb.NonlinearLiftingLine],
)
def test_run_does_not_mutate_vortex_bound_leg(AnalysisClass):
    """
    Regression test: while computing pitching moments, run() used to zero out the
    x-components of `self.vortex_bound_leg` in-place (via an un-copied alias), so any
    post-run consumer of that attribute silently saw corrupted data on swept wings.
    """
    airplane = asb.Airplane(
        wings=[
            asb.Wing(
                symmetric=True,
                xsecs=[
                    asb.WingXSec(xyz_le=[0, 0, 0], chord=1, airfoil=airfoil),
                    asb.WingXSec(
                        xyz_le=[1, 5, 0.2],  # Swept and dihedraled, so that the
                        # bound legs have nonzero x-components.
                        chord=0.6,
                        airfoil=airfoil,
                    ),
                ],
            )
        ],
    )

    analysis = AnalysisClass(
        airplane=airplane,
        op_point=asb.OperatingPoint(velocity=25, alpha=3),
    )
    analysis.run()

    expected_bound_legs = analysis.right_vortex_vertices - analysis.left_vortex_vertices

    assert np.max(np.abs(expected_bound_legs[:, 0])) > 0  # Sanity check: sweep present
    assert np.allclose(analysis.vortex_bound_leg, expected_bound_legs)


if __name__ == "__main__":
    pytest.main([__file__])
