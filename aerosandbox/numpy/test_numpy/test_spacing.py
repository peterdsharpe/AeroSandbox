import aerosandbox.numpy as np
import casadi as cas
import pytest


def test_linspace_matches_numpy_for_casadi_types():
    expected = np.linspace(1, 100, 5)
    result = cas.DM(np.linspace(cas.DM(1), cas.DM(100), 5)).full().flatten()
    assert result == pytest.approx(expected)


def test_logspace_matches_numpy_for_casadi_types():
    expected = np.logspace(0, 2, 5)
    result = cas.DM(np.logspace(cas.DM(0), cas.DM(2), 5)).full().flatten()
    assert result == pytest.approx(expected)


def test_geomspace_matches_numpy_for_casadi_types():
    """np.geomspace() with CasADi inputs should be geometrically spaced
    (regression test: it used to return linearly-spaced values)."""
    expected = np.geomspace(1, 100, 5)  # [1, 3.162..., 10, 31.62..., 100]
    result = cas.DM(np.geomspace(cas.DM(1), cas.DM(100), 5)).full().flatten()
    assert result == pytest.approx(expected)

    with pytest.raises(ValueError):
        np.geomspace(cas.DM(-1), cas.DM(100), 5)


if __name__ == "__main__":
    pytest.main()
