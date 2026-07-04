import pytest

from aerosandbox.library.power_solar import solar_flux


def test_solar_flux_basic_call():
    """Smoke test: a plain call with defaults returns a physically-sensible flux."""
    flux = solar_flux(latitude=40, day_of_year=180, time=3600)
    assert 0 < flux < 1400  # Physically-sensible bounds [W/m^2]


def test_solar_flux_rejects_unknown_kwargs():
    """
    Regression test: solar_flux() used to silently swallow any unrecognized
    keyword argument via **deprecated_kwargs, so a misspelled kwarg (e.g.
    panel_azimut_angle) produced silently-wrong results computed with defaults.
    """
    with pytest.raises(TypeError, match="panel_azimut_angle"):
        solar_flux(
            latitude=40,
            day_of_year=180,
            time=3600,
            panel_azimut_angle=90,  # [sic]: misspelled
        )


def test_solar_flux_warns_on_deprecated_kwargs():
    with pytest.warns(DeprecationWarning, match="scattering"):
        flux_legacy = solar_flux(
            latitude=40,
            day_of_year=180,
            time=3600,
            scattering=True,  # Deprecated: scattering is now always modeled.
        )

    flux = solar_flux(latitude=40, day_of_year=180, time=3600)
    assert flux_legacy == pytest.approx(flux)


if __name__ == "__main__":
    pytest.main([__file__])
