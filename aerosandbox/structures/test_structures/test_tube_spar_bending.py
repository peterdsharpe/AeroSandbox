import aerosandbox as asb
import aerosandbox.numpy as np
import pytest

from aerosandbox.structures.tube_spar_bending import TubeSparBendingStructure


def test_construct_with_default_zero_load():
    """
    Regression test: with the default (zero) distributed force, the variable-scaling
    heuristic used to evaluate to 0, which made Opti.variable() raise
    `ValueError: The 'scale' argument must be a positive number.`
    """
    opti = asb.Opti()
    beam = TubeSparBendingStructure(
        opti=opti,
        length=10,
        diameter_function=0.1,
        wall_thickness_function=1e-3,
    )
    sol = opti.solve(verbose=False)
    beam = sol(beam)

    assert beam.u == pytest.approx(0, abs=1e-6)  # No load -> no deflection


def test_construct_and_solve_with_net_downward_load():
    """
    Regression test: a net-downward (negative) distributed force used to produce a
    negative variable scale, which made Opti.variable() raise ValueError.

    Also checks the solution against the analytic tip deflection of a uniformly-loaded
    cantilever beam, u_tip = q * L^4 / (8 * E * I).
    """
    length = 10
    diameter = 0.1
    wall_thickness = 1e-3
    elastic_modulus = 175e9
    q = -100.0  # Uniform distributed load [N/m], pointing down

    opti = asb.Opti()
    beam = TubeSparBendingStructure(
        opti=opti,
        length=length,
        diameter_function=diameter,
        wall_thickness_function=wall_thickness,
        bending_distributed_force_function=q,
        points_per_point_load=50,
    )
    sol = opti.solve(verbose=False)
    beam = sol(beam)

    moment_of_inertia = np.pi / 8 * diameter**3 * wall_thickness  # Thin-walled tube
    u_tip_analytic = q * length**4 / (8 * elastic_modulus * moment_of_inertia)

    assert beam.u[-1] == pytest.approx(u_tip_analytic, rel=0.01)


def test_solve_with_uniform_upward_load():
    """
    Checks the solution against the analytic tip deflection of a uniformly-loaded
    cantilever beam, u_tip = q * L^4 / (8 * E * I).
    """
    length = 17
    diameter = 0.12
    wall_thickness = 2e-3
    elastic_modulus = 228e9
    q = 150.0  # Uniform distributed load [N/m], pointing up

    opti = asb.Opti()
    beam = TubeSparBendingStructure(
        opti=opti,
        length=length,
        diameter_function=diameter,
        wall_thickness_function=wall_thickness,
        elastic_modulus_function=elastic_modulus,
        bending_distributed_force_function=q,
        points_per_point_load=50,
    )
    sol = opti.solve(verbose=False)
    beam = sol(beam)

    moment_of_inertia = np.pi / 8 * diameter**3 * wall_thickness  # Thin-walled tube
    u_tip_analytic = q * length**4 / (8 * elastic_modulus * moment_of_inertia)

    assert beam.u[-1] == pytest.approx(u_tip_analytic, rel=0.01)


def test_integrate_discrete_intervals_equivalent_to_legacy_trapz():
    """
    TubeSparBendingStructure historically integrated with `asb.numpy.trapz(f) * dy`
    (pending deprecation) and now uses `integrate_discrete_intervals`. This checks
    that the two formulations are numerically identical (bitwise), under both the
    NumPy backend and the CasADi backend.
    """
    import warnings
    import casadi as cas
    import numpy as onp
    from aerosandbox.numpy.integrate_discrete import integrate_discrete_intervals

    rng = onp.random.default_rng(0)
    f = rng.normal(size=50)
    y = onp.sort(rng.uniform(0, 10, size=50))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)

        ### NumPy backend
        legacy = np.trapz(f) * np.diff(y)
        new = integrate_discrete_intervals(f, x=y, method="trapezoidal")
        new_no_dx = integrate_discrete_intervals(
            f, multiply_by_dx=False, method="trapezoidal"
        ) * np.diff(y)
        assert onp.array_equal(legacy, new)
        assert onp.array_equal(legacy, new_no_dx)

        ### CasADi backend
        f_cas = cas.MX(cas.DM(f))
        legacy_cas = onp.array(cas.evalf(np.trapz(f_cas) * np.diff(y))).flatten()
        new_cas = onp.array(
            cas.evalf(integrate_discrete_intervals(f_cas, x=y, method="trapezoidal"))
        ).flatten()
        assert onp.array_equal(legacy_cas, new_cas)
        assert onp.array_equal(legacy, new_cas)


def test_volume_and_total_force_match_legacy_trapz_formulation():
    """
    Checks TubeSparBendingStructure.volume() and .total_force() (post-migration to
    integrate_discrete_intervals) against manual trapezoidal integration, for both
    the thin-tube and thick-tube formulations, and against analytic values.
    """
    import numpy as onp

    length = 10
    diameter = 0.1
    wall_thickness = 5e-3
    q = 100.0

    for assume_thin_tube in [True, False]:
        opti = asb.Opti()
        beam = TubeSparBendingStructure(
            opti=opti,
            length=length,
            diameter_function=diameter,
            wall_thickness_function=wall_thickness,
            bending_distributed_force_function=q,
            assume_thin_tube=assume_thin_tube,
        )
        sol = opti.solve(verbose=False)
        beam = sol(beam)

        ### Manual trapezoidal integration (the legacy `trapz(f) * dy` formulation)
        def manual_integral(f):
            return onp.sum((f[1:] + f[:-1]) / 2 * onp.diff(beam.y))

        if assume_thin_tube:
            section_area = onp.pi * beam.diameter * beam.wall_thickness
            section_area_analytic = onp.pi * diameter * wall_thickness
        else:
            section_area = (
                onp.pi
                / 4
                * (
                    (beam.diameter + beam.wall_thickness) ** 2
                    - (beam.diameter - beam.wall_thickness) ** 2
                )
            )
            section_area_analytic = (
                onp.pi
                / 4
                * ((diameter + wall_thickness) ** 2 - (diameter - wall_thickness) ** 2)
            )

        assert beam.volume() == pytest.approx(manual_integral(section_area), rel=1e-12)
        assert beam.volume() == pytest.approx(section_area_analytic * length, rel=1e-12)

        assert beam.total_force() == pytest.approx(
            manual_integral(beam.distributed_force), rel=1e-12
        )
        assert beam.total_force() == pytest.approx(q * length, rel=1e-12)


if __name__ == "__main__":
    pytest.main([__file__])
