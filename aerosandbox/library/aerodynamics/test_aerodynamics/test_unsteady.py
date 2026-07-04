import aerosandbox.numpy as np
import pytest

from aerosandbox.library.aerodynamics.unsteady import (
    calculate_reduced_time,
    wagners_function,
    kussners_function,
    indicial_gust_response,
    calculate_lift_due_to_transverse_gust,
    calculate_lift_due_to_pitching_profile,
    pitching_through_transverse_gust,
    top_hat_gust,
)


def test_indicial_gust_response_is_chord_invariant():
    """
    Reduced time is already nondimensionalized by the semichord, so the gust
    response at a given reduced time must not depend on the (dimensional) chord.
    """
    reduced_time = np.linspace(0, 10, 21)
    kwargs = dict(
        gust_velocity=1.0,
        plate_velocity=10.0,
        angle_of_attack=20,  # degrees; nonzero to activate the gust-entry offset
    )
    cl_chord_1 = indicial_gust_response(reduced_time, **kwargs, chord=1)
    cl_chord_4 = indicial_gust_response(reduced_time, **kwargs, chord=4)

    assert cl_chord_1 == pytest.approx(cl_chord_4)


def test_indicial_gust_response_offset_in_semichords():
    """
    The gust-entry offset should equal (1 - cos(alpha)) semichords, so the
    response should equal Kussner's function evaluated at the shifted reduced time.
    """
    reduced_time = np.linspace(0, 10, 21)
    gust_velocity = 1.0
    plate_velocity = 10.0
    alpha_deg = 20.0
    alpha_rad = np.deg2rad(alpha_deg)

    cl = indicial_gust_response(
        reduced_time,
        gust_velocity=gust_velocity,
        plate_velocity=plate_velocity,
        angle_of_attack=alpha_deg,
        chord=3.7,  # Should have no effect
    )
    cl_expected = (
        2
        * np.pi
        * np.arctan(gust_velocity / plate_velocity)
        * np.cos(alpha_rad)
        * kussners_function(reduced_time - (1 - np.cos(alpha_rad)))
    )
    assert cl == pytest.approx(cl_expected)


def test_indicial_gust_response_zero_alpha_unchanged():
    """At zero angle of attack, there is no offset, for any chord."""
    reduced_time = np.linspace(0, 10, 21)
    cl = indicial_gust_response(
        reduced_time, gust_velocity=1.0, plate_velocity=10.0, chord=5
    )
    cl_expected = 2 * np.pi * np.arctan(1.0 / 10.0) * kussners_function(reduced_time)
    assert cl == pytest.approx(cl_expected)


def test_transverse_gust_lift_is_chord_invariant():
    reduced_time = np.linspace(0, 10, 5)
    kwargs = dict(
        gust_velocity_profile=top_hat_gust,
        plate_velocity=10.0,
        angle_of_attack=20.0,  # degrees, constant
    )
    cl_chord_1 = calculate_lift_due_to_transverse_gust(reduced_time, **kwargs, chord=1)
    cl_chord_4 = calculate_lift_due_to_transverse_gust(reduced_time, **kwargs, chord=4)

    assert cl_chord_1 == pytest.approx(cl_chord_4)


def test_pitching_through_transverse_gust_accepts_float_angle_of_attack():
    """
    The docstring advertises `angle_of_attack: Callable | float`, but float
    input used to crash with `TypeError: 'float' object is not callable`.
    """
    reduced_time = np.linspace(0, 10, 5)

    cl_float = pitching_through_transverse_gust(
        reduced_time,
        gust_velocity_profile=top_hat_gust,
        plate_velocity=10.0,
        angle_of_attack=5.0,
    )

    ### Should be identical to passing an equivalent constant Callable
    cl_callable = pitching_through_transverse_gust(
        reduced_time,
        gust_velocity_profile=top_hat_gust,
        plate_velocity=10.0,
        angle_of_attack=lambda s: 5.0,
    )
    assert cl_float == pytest.approx(cl_callable)


if __name__ == "__main__":
    pytest.main([__file__])
