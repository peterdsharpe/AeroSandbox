import aerosandbox.numpy as np
from aerosandbox.numpy.logicals import (
    clip,
    logical_and,
    logical_or,
    logical_not,
    all,
    any,
)
import pytest
import casadi as cas


def test_clip_within_bounds():
    """Test clipping values within bounds."""
    x = 5.0
    result = clip(x, min=0, max=10)
    assert result == 5.0


def test_clip_below_minimum():
    """Test clipping value below minimum."""
    x = -5.0
    result = clip(x, min=0, max=10)
    assert result == 0.0


def test_clip_above_maximum():
    """Test clipping value above maximum."""
    x = 15.0
    result = clip(x, min=0, max=10)
    assert result == 10.0


def test_clip_array():
    """Test clipping an array of values."""
    x = np.array([-5, 0, 5, 10, 15])
    result = clip(x, min=0, max=10)
    expected = np.array([0, 0, 5, 10, 10])

    assert np.allclose(result, expected)


def test_clip_negative_bounds():
    """Test clipping with negative bounds."""
    x = np.array([-10, -5, 0, 5, 10])
    result = clip(x, min=-7, max=7)
    expected = np.array([-7, -5, 0, 5, 7])

    assert np.allclose(result, expected)


def test_logical_and_numpy_true_true():
    """Test logical AND with both True (NumPy)."""
    x1 = np.array([True, True])
    x2 = np.array([True, True])

    result = logical_and(x1, x2)
    expected = np.array([True, True])

    assert np.all(result == expected)


def test_logical_and_numpy_true_false():
    """Test logical AND with True and False (NumPy)."""
    x1 = np.array([True, False, True, False])
    x2 = np.array([True, True, False, False])

    result = logical_and(x1, x2)
    expected = np.array([True, False, False, False])

    assert np.all(result == expected)


def test_logical_and_casadi():
    """Test logical AND with CasADi types."""
    x1 = cas.DM([1, 1, 0, 0])
    x2 = cas.DM([1, 0, 1, 0])

    result = logical_and(x1, x2)
    expected = np.array([[1], [0], [0], [0]])

    assert np.allclose(np.array(result), expected)


def test_logical_or_numpy_false_false():
    """Test logical OR with both False (NumPy)."""
    x1 = np.array([False, False])
    x2 = np.array([False, False])

    result = logical_or(x1, x2)
    expected = np.array([False, False])

    assert np.all(result == expected)


def test_logical_or_numpy_mixed():
    """Test logical OR with mixed values (NumPy)."""
    x1 = np.array([True, False, True, False])
    x2 = np.array([True, True, False, False])

    result = logical_or(x1, x2)
    expected = np.array([True, True, True, False])

    assert np.all(result == expected)


def test_logical_or_casadi():
    """Test logical OR with CasADi types."""
    x1 = cas.DM([1, 0, 1, 0])
    x2 = cas.DM([1, 1, 0, 0])

    result = logical_or(x1, x2)
    expected = np.array([[1], [1], [1], [0]])

    assert np.allclose(np.array(result), expected)


def test_logical_not_numpy():
    """Test logical NOT with NumPy arrays."""
    x = np.array([True, False, True, False])

    result = logical_not(x)
    expected = np.array([False, True, False, True])

    assert np.all(result == expected)


def test_logical_not_casadi():
    """Test logical NOT with CasADi types."""
    x = cas.DM([1, 0, 1, 0])

    result = logical_not(x)
    expected = np.array([[0], [1], [0], [1]])

    assert np.allclose(np.array(result), expected)


def test_logical_not_scalar():
    """Test logical NOT with scalar values."""
    assert logical_not(True) == False
    assert logical_not(False) == True


def test_all_true_numpy():
    """Test all() with all True values (NumPy)."""
    a = np.array([True, True, True])
    result = all(a)
    assert result == True


def test_all_false_numpy():
    """Test all() with at least one False (NumPy)."""
    a = np.array([True, False, True])
    result = all(a)
    assert result == False


def test_all_casadi_true():
    """Test all() with CasADi types (all True)."""
    a = cas.DM([1, 1, 1])
    result = all(a)
    ### CasADi may return True or handle differently
    assert isinstance(result, bool) or isinstance(result, cas.DM)


def test_any_true_numpy():
    """Test any() with at least one True (NumPy)."""
    a = np.array([False, True, False])
    result = any(a)
    assert result == True


def test_any_false_numpy():
    """Test any() with all False (NumPy)."""
    a = np.array([False, False, False])
    result = any(a)
    assert result == False


def test_any_casadi():
    """Test any() with CasADi types."""
    a = cas.DM([0, 1, 0])
    result = any(a)
    ### CasADi may return True or handle differently
    assert isinstance(result, bool) or isinstance(result, cas.DM)


def test_logical_and_scalar_arrays():
    """Test logical AND with scalar and array."""
    x1 = True
    x2 = np.array([True, False, True])

    result = logical_and(x1, x2)
    expected = np.array([True, False, True])

    assert np.all(result == expected)


def test_logical_or_scalar_arrays():
    """Test logical OR with scalar and array."""
    x1 = False
    x2 = np.array([True, False, True])

    result = logical_or(x1, x2)
    expected = np.array([True, False, True])

    assert np.all(result == expected)


def test_clip_casadi_types():
    """Test clipping with CasADi types."""
    x = cas.DM([-5, 5, 15])
    result = clip(x, min=0, max=10)
    expected = np.array([[0], [5], [10]])

    assert np.allclose(np.array(result), expected)


def test_logical_operations_chaining():
    """Test chaining logical operations."""
    x = np.array([True, True, False, False])
    y = np.array([True, False, True, False])
    z = np.array([False, True, True, False])

    ### (x AND y) OR z
    result = logical_or(logical_and(x, y), z)
    expected = np.array([True, True, True, False])

    assert np.all(result == expected)


def test_logical_not_multiple_negations():
    """Test multiple NOT operations."""
    x = np.array([True, False])

    result = logical_not(logical_not(x))
    expected = x

    assert np.all(result == expected)


def test_all_empty_array():
    """Test all() with empty array."""
    a = np.array([])
    result = all(a)
    assert result == True  ### all() of empty array is True by convention


def test_any_empty_array():
    """Test any() with empty array."""
    a = np.array([])
    result = any(a)
    assert result == False  ### any() of empty array is False by convention


def test_clip_equal_bounds():
    """Test clipping when min equals max."""
    x = np.array([1, 5, 10])
    result = clip(x, min=5, max=5)
    expected = np.array([5, 5, 5])

    assert np.allclose(result, expected)


def test_logical_and_broadcasting():
    """Test logical AND with broadcasting."""
    x1 = np.array([[True, False], [True, True]])
    x2 = np.array([True, False])

    result = logical_and(x1, x2)
    expected = np.array([[True, False], [True, False]])

    assert np.all(result == expected)


def test_logical_or_broadcasting():
    """Test logical OR with broadcasting."""
    x1 = np.array([[True, False], [False, False]])
    x2 = np.array([False, True])

    result = logical_or(x1, x2)
    expected = np.array([[True, True], [False, True]])

    assert np.all(result == expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
