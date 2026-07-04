import pytest

from aerosandbox.library.propulsion_small_solid_rocket import thrust_coefficient


def test_thrust_coefficient_defaults():
    """
    Regression test: thrust_coefficient() used to raise TypeError when p_a and
    er were left at their default of None, despite the docstring promising
    matched-expansion behavior in that case.
    """
    C_F = thrust_coefficient(2e6, 1e5, 1.2)
    assert C_F == pytest.approx(1.4084408756881703)

    # Matched expansion (p_a == exit_pressure) must equal the default case,
    # since the pressure-thrust term is zero.
    C_F_matched = thrust_coefficient(2e6, 1e5, 1.2, p_a=1e5, er=5.0)
    assert C_F_matched == pytest.approx(C_F)


def test_thrust_coefficient_pressure_thrust_term():
    C_F_matched = thrust_coefficient(2e6, 1e5, 1.2)

    # Underexpanded nozzle (p_a < exit_pressure): pressure thrust adds.
    C_F_under = thrust_coefficient(2e6, 1e5, 1.2, p_a=5e4, er=5.0)
    assert C_F_under == pytest.approx(C_F_matched + 5.0 * (1e5 - 5e4) / 2e6)


def test_thrust_coefficient_rejects_partial_arguments():
    with pytest.raises(ValueError):
        thrust_coefficient(2e6, 1e5, 1.2, p_a=1e5)
    with pytest.raises(ValueError):
        thrust_coefficient(2e6, 1e5, 1.2, er=5.0)


def test_thrust_coefficient_casadi():
    casadi = pytest.importorskip("casadi")
    chamber_pressure = casadi.MX.sym("chamber_pressure")
    out = thrust_coefficient(chamber_pressure, 1e5, 1.2)
    assert isinstance(out, casadi.MX)
    out = thrust_coefficient(chamber_pressure, 1e5, 1.2, p_a=5e4, er=5.0)
    assert isinstance(out, casadi.MX)


if __name__ == "__main__":
    pytest.main([__file__])
