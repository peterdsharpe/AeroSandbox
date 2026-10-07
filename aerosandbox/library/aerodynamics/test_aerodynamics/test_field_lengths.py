import aerosandbox as asb
import pytest
from aerosandbox.tools import units as u

from aerosandbox.library.field_lengths import (
    field_length_analysis,
    field_length_analysis_torenbeek,
)


def test_field_length_analysis_torenbeek_reasonable_values():
    results = field_length_analysis_torenbeek(
        design_mass_TOGW=19000 * u.lbm,
        thrust_at_liftoff=19000 * u.lbf * 0.3,
        lift_over_drag_climb=20,
        CL_max=1.9,
        s_ref=24,
        n_engines=2,
    )

    for key in [
        "takeoff_ground_roll_distance",
        "takeoff_airborne_distance",
        "takeoff_total_distance",
        "balanced_field_length",
        "landing_airborne_distance",
        "landing_ground_roll_distance",
        "landing_total_distance",
        "V_stall",
        "V_liftoff",
        "V_obstacle",
        "V_approach",
        "V_touchdown",
        "flight_path_angle_climb",
    ]:
        assert results[key] > 0, f"{key} should be positive"

    ### BFL should never be shorter than the normal takeoff distance
    assert results["balanced_field_length"] >= results["takeoff_total_distance"]

    ### Speeds should be ordered sensibly
    assert results["V_stall"] < results["V_liftoff"] <= results["V_obstacle"]


def test_field_length_analysis_torenbeek_cannot_climb_raises():
    """
    If thrust_over_weight <= 1 / (L/D), the climb angle is non-positive; this
    used to produce ZeroDivisionError (or a confusing negative-softness error)
    instead of an informative message.
    """
    with pytest.raises(ValueError, match="climb"):
        field_length_analysis_torenbeek(
            design_mass_TOGW=1000,
            thrust_at_liftoff=1000 * 9.81 * 0.05,  # T/W = 0.05 == 1 / (L/D)
            lift_over_drag_climb=20,
            CL_max=1.9,
            s_ref=10,
            n_engines=2,
        )

    with pytest.raises(ValueError, match="climb"):
        field_length_analysis_torenbeek(
            design_mass_TOGW=1000,
            thrust_at_liftoff=1000 * 9.81 * 0.01,  # T/W well below 1 / (L/D)
            lift_over_drag_climb=20,
            CL_max=1.9,
            s_ref=10,
            n_engines=2,
        )


def test_field_length_analysis_torenbeek_casadi_symbolics():
    """The analysis should remain usable inside an asb.Opti optimization problem."""
    opti = asb.Opti()
    thrust = opti.variable(init_guess=30000, lower_bound=1000)
    results = field_length_analysis_torenbeek(
        design_mass_TOGW=8000,
        thrust_at_liftoff=thrust,
        lift_over_drag_climb=15,
        CL_max=1.9,
        s_ref=24,
        n_engines=2,
    )
    opti.subject_to(results["takeoff_total_distance"] < 1000)
    opti.minimize(thrust)
    sol = opti.solve(verbose=False)
    assert sol(thrust) > 0


def test_field_length_analysis_reasonable_values():
    results = field_length_analysis(
        design_mass_TOGW=19000 * u.lbm,
        thrust_at_liftoff=19000 * u.lbf * 0.3,
        lift_over_drag_climb=20,
        CL_max=1.9,
        s_ref=24,
        n_engines=2,
        V_engine_failure_balanced_field_length=70,
    )

    for key in [
        "takeoff_ground_roll_distance",
        "takeoff_airborne_distance",
        "takeoff_total_distance",
        "balanced_field_length_accept",
        "balanced_field_length_reject",
        "landing_airborne_distance",
        "landing_ground_roll_distance",
        "landing_total_distance",
        "V_stall",
        "V_liftoff",
        "V_touchdown",
        "flight_path_angle_climb",
    ]:
        assert results[key] > 0, f"{key} should be positive"


if __name__ == "__main__":
    pytest.main([__file__])
