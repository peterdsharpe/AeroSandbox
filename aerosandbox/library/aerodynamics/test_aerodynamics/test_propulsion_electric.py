import pytest

import aerosandbox as asb
from aerosandbox.library.propulsion_electric import (
    electric_propeller_propulsion_analysis,
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


def test_electric_propeller_propulsion_analysis_documented_keys():
    """
    The function returns locals(); check that the keys called out in the
    docstring are actually present.
    """
    result = electric_propeller_propulsion_analysis(
        total_thrust=1000,
        n_engines=2,
        propeller_diameter=2.0,
        op_point=asb.OperatingPoint(velocity=50),
        motor_kv=100,
        motor_no_load_current=1,
        motor_resistance=0.05,
        wire_resistance=0.01,
        battery_voltage=400,
    )
    for key in [
        "air_power",
        "shaft_power",
        "motor_electrical_power",
        "esc_electrical_power",
        "battery_power",
        "battery_current",
        "propeller_efficiency",
        "motor_efficiency",
        "wire_efficiency",
        "overall_efficiency",
    ]:
        assert key in result
        assert result[key] > 0


if __name__ == "__main__":
    pytest.main([__file__])
