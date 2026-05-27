import aerosandbox as asb
import aerosandbox.numpy as np
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
