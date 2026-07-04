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


if __name__ == "__main__":
    pytest.main([__file__])
