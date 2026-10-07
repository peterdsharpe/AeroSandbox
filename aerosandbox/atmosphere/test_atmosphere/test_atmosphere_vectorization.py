import casadi
import pytest

import aerosandbox.numpy as np
from aerosandbox import Atmosphere


def test_len_scalar():
    assert len(Atmosphere(altitude=1000.0)) == 1


def test_len_vector():
    assert len(Atmosphere(altitude=np.array([1e3, 2e3, 3e3]))) == 3


def test_len_length_1_array():
    assert len(Atmosphere(altitude=np.array([1e3]))) == 1


def test_len_with_trailing_length_1_array():
    ### Regression test: a subscriptable length-1 attribute used to reset the
    ### detected vector length back to 1.
    atmo = Atmosphere(
        altitude=np.array([1e3, 2e3, 3e3]),
        temperature_deviation=np.array([5.0]),
    )
    assert len(atmo) == 3


def test_len_with_leading_length_1_array():
    atmo = Atmosphere(
        altitude=np.array([1e3]),
        temperature_deviation=np.array([1.0, 2.0, 3.0]),
    )
    assert len(atmo) == 3


def test_len_with_mismatched_lengths_raises():
    atmo = Atmosphere(
        altitude=np.array([1e3, 2e3, 3e3]),
        temperature_deviation=np.array([1.0, 2.0]),
    )
    with pytest.raises(ValueError):
        len(atmo)


def test_len_casadi():
    assert len(Atmosphere(altitude=casadi.DM([1e3, 2e3, 3e3]))) == 3
    assert (
        len(
            Atmosphere(
                altitude=casadi.DM([1e3, 2e3, 3e3]),
                temperature_deviation=casadi.DM([5.0]),
            )
        )
        == 3
    )
    assert len(Atmosphere(altitude=casadi.MX.sym("alt", 3, 1))) == 3


def test_getitem_with_trailing_length_1_array():
    atmo = Atmosphere(
        altitude=np.array([1e3, 2e3, 3e3]),
        temperature_deviation=np.array([5.0]),
    )
    atmo_1 = atmo[1]
    assert atmo_1.altitude == pytest.approx(2e3)
    assert atmo_1.temperature_deviation == pytest.approx(5.0)


if __name__ == "__main__":
    pytest.main([__file__])
