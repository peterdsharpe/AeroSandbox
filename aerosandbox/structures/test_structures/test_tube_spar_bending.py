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

    I = np.pi / 8 * diameter**3 * wall_thickness  # Thin-walled tube
    u_tip_analytic = q * length**4 / (8 * elastic_modulus * I)

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

    I = np.pi / 8 * diameter**3 * wall_thickness  # Thin-walled tube
    u_tip_analytic = q * length**4 / (8 * elastic_modulus * I)

    assert beam.u[-1] == pytest.approx(u_tip_analytic, rel=0.01)


if __name__ == "__main__":
    pytest.main([__file__])
