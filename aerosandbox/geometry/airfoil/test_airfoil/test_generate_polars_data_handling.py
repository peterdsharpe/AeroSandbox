"""
Tests for Airfoil.generate_polars() data handling (cache pathing and symmetric-polar mirroring).

These tests monkeypatch asb.XFoil with a synthetic-data stand-in, so they do not require the xfoil binary.
"""

import aerosandbox as asb
import aerosandbox.numpy as np
import numpy as onp
import pytest


class FakeXFoil:
    """A stand-in for asb.XFoil that returns plausible synthetic polar data."""

    def __init__(self, airfoil, Re, **kwargs):
        self.airfoil = airfoil
        self.Re = Re

    def alpha(self, alpha):
        alpha = onp.reshape(onp.asarray(alpha, dtype=float), -1)
        return {
            "alpha": alpha,
            "CL": 0.11 * alpha,
            "CD": 0.01 + 2e-4 * alpha**2,
            "CDp": 0.005 + 1e-4 * alpha**2,
            "CM": -0.002 * alpha,
            "Cpmin": -1 - 0.1 * onp.abs(alpha),
            "Xcpmin": 0.3 * onp.ones_like(alpha),
            "Chinge": 0.01 * alpha,
            "Top_Xtr": 0.7 - 0.02 * alpha,
            "Bot_Xtr": 0.7 + 0.02 * alpha,
        }


@pytest.fixture
def fake_xfoil(monkeypatch):
    import aerosandbox.aerodynamics.aero_2D as aero_2D

    monkeypatch.setattr(aero_2D, "XFoil", FakeXFoil)


def test_generate_polars_cache_filename_without_directory(
    fake_xfoil, tmp_path, monkeypatch
):
    """
    A cache_filename with no directory component (i.e., a bare filename in the current directory)
    should work, rather than crashing in os.makedirs('').
    """
    monkeypatch.chdir(tmp_path)

    af = asb.Airfoil("naca0012")
    af.generate_polars(
        alphas=np.linspace(-5, 5, 6),
        Res=np.geomspace(1e5, 1e6, 2),
        cache_filename="polars_cache.json",
    )

    assert (tmp_path / "polars_cache.json").exists()


if __name__ == "__main__":
    pytest.main([__file__])
