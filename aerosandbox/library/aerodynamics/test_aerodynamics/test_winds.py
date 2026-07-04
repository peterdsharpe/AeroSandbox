import warnings
from pathlib import Path

import numpy as np
import pytest

from aerosandbox.library import winds


def test_module_compiles_without_syntax_warnings():
    """
    Docstrings with un-escaped Windows paths used to emit
    'SyntaxWarning: invalid escape sequence' on Python 3.12+.
    """
    source_path = Path(winds.__file__)
    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        compile(source_path.read_text(), str(source_path), "exec")


def test_wind_speed_conus_summer_99():
    ### Jet-stream altitudes over the central US should see strong winds...
    wind_speed_jet_stream = winds.wind_speed_conus_summer_99(
        altitude=12000, latitude=38
    )
    assert 20 < wind_speed_jet_stream < 100

    ### ...much stronger than near the surface.
    wind_speed_surface = winds.wind_speed_conus_summer_99(altitude=0, latitude=38)
    assert 0 < wind_speed_surface < wind_speed_jet_stream


def test_wind_speed_world_95_scalar():
    wind_speed = winds.wind_speed_world_95(altitude=12000, latitude=45, day_of_year=180)
    assert 0 < wind_speed < 100
    assert np.isfinite(wind_speed)


def test_wind_speed_world_95_vectorized():
    wind_speeds = winds.wind_speed_world_95(
        altitude=np.array([10000.0, 15000.0]),
        latitude=np.array([40.0, 50.0]),
        day_of_year=np.array([100.0, 200.0]),
    )
    assert np.shape(wind_speeds) == (2,)
    assert np.all(wind_speeds > 0)
    assert np.all(wind_speeds < 150)


def test_wind_speed_world_95_day_of_year_wraps_smoothly():
    """The seasonal model is periodic; Dec 31 and Jan 0 should roughly agree."""
    wind_end_of_year = winds.wind_speed_world_95(
        altitude=12000, latitude=45, day_of_year=365
    )
    wind_start_of_year = winds.wind_speed_world_95(
        altitude=12000, latitude=45, day_of_year=0
    )
    assert wind_end_of_year == pytest.approx(wind_start_of_year, rel=0.05)


def test_tropopause_altitude():
    ### Tropopause altitudes should be physically sensible (roughly 7-20 km)
    trop_alt_equator = winds.tropopause_altitude(latitude=0, day_of_year=0)
    trop_alt_arctic = winds.tropopause_altitude(latitude=70, day_of_year=0)

    assert 7e3 < trop_alt_arctic < trop_alt_equator < 20e3


def test_tropopause_altitude_vectorized():
    trop_alts = winds.tropopause_altitude(
        latitude=np.array([0.0, 45.0, 70.0]),
        day_of_year=np.array([0.0, 180.0, 300.0]),
    )
    assert np.shape(trop_alts) == (3,)
    assert np.all(trop_alts > 7e3)
    assert np.all(trop_alts < 20e3)


if __name__ == "__main__":
    pytest.main([__file__])
