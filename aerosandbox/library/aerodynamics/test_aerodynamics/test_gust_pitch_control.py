import numpy as onp
import pytest

import aerosandbox.numpy as np
from aerosandbox.library.aerodynamics.unsteady import calculate_reduced_time
from aerosandbox.library.gust_pitch_control import TransverseGustPitchControl


def test_transverse_gust_pitch_control_constructs_and_solves():
    """
    Regression test: TransverseGustPitchControl used to crash on construction
    with a TypeError, because its governing equations called np.sum() on a
    generator expression (which NumPy rejects and which the dual-backend
    asarray() cannot convert).
    """
    N = 15
    time = np.linspace(0, 5, N)
    wing_velocity = 2.0
    chord = 2.0
    reduced_time = calculate_reduced_time(time, wing_velocity, chord)
    gust_profile = 0.1 * np.sin(2 * np.pi * time / 5)  # smooth gust

    optimal = TransverseGustPitchControl(  # crashed before the fix
        reduced_time=reduced_time,
        gust_profile=gust_profile,
        velocity=wing_velocity,
    )

    optimal.calculate_transients()  # also crashed before the fix (same pattern)

    assert onp.all(onp.isfinite(optimal.optimal_pitching_profile_rad))
    assert onp.all(onp.isfinite(optimal.optimal_lift_history))
    assert onp.all(onp.isfinite(optimal.pitching_lift))
    assert onp.all(onp.isfinite(optimal.gust_lift))
    assert onp.all(onp.isfinite(optimal.added_mass_lift))

    # The optimizer should hold the initial angle of attack at zero, per the constraint.
    assert optimal.optimal_pitching_profile_rad[0] == pytest.approx(0, abs=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
