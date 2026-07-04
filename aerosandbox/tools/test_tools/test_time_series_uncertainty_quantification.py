import numpy as np
import pytest

import aerosandbox.tools.statistics.time_series_uncertainty_quantification as tsuq


def test_bootstrap_fits_shapes():
    np.random.seed(0)
    x = np.linspace(0, 1, 50)
    y = np.sin(3 * x) + 0.01 * np.random.randn(len(x))

    x_fit, y_bootstrap_fits = tsuq.bootstrap_fits(
        x,
        y,
        y_noise_stdev=0.01,
        n_bootstraps=10,
        fit_points=20,
    )
    assert x_fit.shape == (20,)
    assert y_bootstrap_fits.shape == (10, 20)


def test_bootstrap_fits_returns_splines_when_fit_points_is_none():
    np.random.seed(0)
    x = np.linspace(0, 1, 50)
    y = np.sin(3 * x) + 0.01 * np.random.randn(len(x))

    splines = tsuq.bootstrap_fits(
        x,
        y,
        y_noise_stdev=0.01,
        n_bootstraps=3,
        fit_points=None,
        normalize=False,
    )
    assert len(splines) == 3
    for spline in splines:
        assert np.isfinite(spline(0.5))


def test_bootstrap_fits_raises_instead_of_hanging_on_persistent_nan(monkeypatch):
    """
    If every spline fit evaluates to NaN, bootstrap_fits should raise a clear
    ValueError instead of looping forever.
    """

    class NaNSpline:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, x):
            return np.nan

    monkeypatch.setattr(tsuq, "Spline", NaNSpline)

    x = np.linspace(0, 1, 20)
    y = x**2

    with pytest.raises(ValueError, match="non-NaN"):
        tsuq.bootstrap_fits(
            x,
            y,
            y_noise_stdev=0.1,
            n_bootstraps=5,
        )


if __name__ == "__main__":
    pytest.main([__file__])
