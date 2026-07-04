import pytest

from aerosandbox.library.aerodynamics.inviscid import oswalds_efficiency


def test_oswalds_efficiency_valid_methods():
    for method in ["nita_scholz", "kroo"]:
        e = oswalds_efficiency(
            taper_ratio=0.5,
            aspect_ratio=8,
            sweep=10,
            method=method,
        )
        assert 0 < e <= 1


def test_oswalds_efficiency_invalid_method_raises_value_error():
    """Used to raise UnboundLocalError instead of ValueError."""
    with pytest.raises(ValueError):
        oswalds_efficiency(
            taper_ratio=0.5,
            aspect_ratio=8,
            method="raymer",
        )


if __name__ == "__main__":
    pytest.main([__file__])
