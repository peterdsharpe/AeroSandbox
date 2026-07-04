import warnings

import aerosandbox as asb
import numpy as onp
import pytest


def test_eq_with_shape_mismatched_array_attributes():
    """
    Comparing two AeroSandbox objects whose array attributes have different
    shapes should return False, not raise a ValueError.

    (With NumPy 2.x, `==` on non-broadcastable arrays raises ValueError.)
    """
    mp1 = asb.MassProperties(mass=onp.array([1.0, 2.0, 3.0]))
    mp2 = asb.MassProperties(mass=onp.array([1.0, 2.0]))

    assert (mp1 == mp2) is False
    assert (mp2 == mp1) is False
    assert (mp1 != mp2) is True


def test_eq_with_equal_and_unequal_array_attributes():
    mp1 = asb.MassProperties(mass=onp.array([1.0, 2.0]))
    mp2 = asb.MassProperties(mass=onp.array([1.0, 2.0]))
    mp3 = asb.MassProperties(mass=onp.array([1.0, 3.0]))

    assert mp1 == mp2
    assert mp1 != mp3


def test_array_protocol_accepts_copy_keyword():
    """
    NumPy 2.0's `__array__` protocol requires `dtype` and `copy` keyword
    arguments; without them, NumPy emits a DeprecationWarning (which will
    eventually become a TypeError).
    """
    mp = asb.MassProperties(mass=1.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Any DeprecationWarning -> failure
        arr = onp.asarray(mp)
        arr_copied = onp.asarray(mp, copy=True)

    for a in [arr, arr_copied]:
        assert a.shape == ()
        assert a.dtype == onp.dtype("O")
        assert a[()] is mp

    ### A no-copy conversion is impossible, and should say so per the protocol:
    with pytest.raises(ValueError):
        onp.asarray(mp, copy=False)


def test_len_error_message_for_mismatched_vectorized_lengths():
    mp = asb.MassProperties(
        mass=onp.array([1.0, 2.0, 3.0]),
        x_cg=onp.array([0.0, 1.0]),
    )
    with pytest.raises(ValueError, match="State variables appear vectorized"):
        len(mp)


def test_generate_possible_set_of_point_masses_roundtrip():
    """
    Regression test for the internal length-scale estimate (radius of
    gyration is sqrt(I / m), not sqrt(I) / m): the generated point masses
    must recombine into the original mass properties, including for masses
    far from 1 kg.
    """
    mp = asb.MassProperties(
        mass=500.0,
        x_cg=1.0,
        y_cg=-0.5,
        z_cg=0.25,
        Ixx=20.0,
        Iyy=30.0,
        Izz=40.0,
        Ixy=1.0,
        Iyz=-2.0,
        Ixz=3.0,
    )
    point_masses = mp.generate_possible_set_of_point_masses()

    for pm in point_masses:
        assert pm.is_point_mass()

    assert mp.allclose(sum(point_masses), rtol=1e-4, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
