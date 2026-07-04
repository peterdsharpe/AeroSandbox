import pytest

from aerosandbox.library.propulsion_electric import (
    mass_motor_electric,
)


def test_mass_motor_electric_known_methods():
    assert mass_motor_electric(1000, method="burton") == pytest.approx(
        0.24224806201550386
    )
    assert mass_motor_electric(1000, method="hobbyking") == pytest.approx(
        0.2025349606824973
    )
    assert mass_motor_electric(
        1000, kv_rpm_volt=1000, voltage=20, method="astroflight"
    ) == pytest.approx(0.49119999999999997)


def test_mass_motor_electric_bad_method_raises():
    """
    Regression test: an unrecognized `method` used to fall off the end of the
    if/elif chain and silently return None.
    """
    with pytest.raises(ValueError, match="method"):
        mass_motor_electric(1000, method="hobby_king")  # [sic]: typo'd method


if __name__ == "__main__":
    pytest.main([__file__])
