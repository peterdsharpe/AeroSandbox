from aerosandbox.numpy.determine_type import *
import pytest
import numpy as np
import casadi as cas


def test_int():
    assert not is_casadi_type(5, recursive=True)
    assert not is_casadi_type(5, recursive=False)


def test_float():
    assert not is_casadi_type(5.0, recursive=True)
    assert not is_casadi_type(5.0, recursive=False)


def test_numpy():
    assert not is_casadi_type(np.array([1, 2, 3]), recursive=True)
    assert not is_casadi_type(np.array([1, 2, 3]), recursive=False)


def test_casadi():
    assert is_casadi_type(cas.MX(np.ones(5)), recursive=False)
    assert is_casadi_type(cas.MX(np.ones(5)), recursive=True)


def test_numpy_list():
    assert not is_casadi_type([np.array(5), np.array(7)], recursive=False)
    assert not is_casadi_type([np.array(5), np.array(7)], recursive=True)


def test_casadi_list():
    assert not (
        is_casadi_type([cas.MX(np.ones(5)), cas.MX(np.ones(5))], recursive=False)
    )
    assert (
        is_casadi_type([cas.MX(np.ones(5)), cas.MX(np.ones(5))], recursive=True)
    )


def test_mixed_list():
    assert not is_casadi_type([np.array(5), cas.MX(np.ones(5))], recursive=False)
    assert is_casadi_type([np.array(5), cas.MX(np.ones(5))], recursive=True)


def test_multi_level_contaminated_list():
    a = [[1 for _ in range(10)] for _ in range(10)]

    assert not is_casadi_type(a, recursive=False)
    assert not is_casadi_type(a, recursive=True)

    a[5][5] = cas.MX(1)

    assert not is_casadi_type(a, recursive=False)
    assert is_casadi_type(a, recursive=True)

    a[5][5] = np.array(cas.DM(1), dtype="O")

    assert not is_casadi_type(a, recursive=False)
    assert not is_casadi_type(a, recursive=True)


if __name__ == "__main__":
    pytest.main([__file__])
