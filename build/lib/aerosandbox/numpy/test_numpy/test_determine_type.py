from aerosandbox.numpy.determine_type import *
import pytest
import numpy as np
import casadi as cas


def test_int():
    assert is_casadi_type(5, recursive=True) == False
    assert is_casadi_type(5, recursive=False) == False


def test_float():
    assert is_casadi_type(5., recursive=True) == False
    assert is_casadi_type(5., recursive=False) == False


def test_numpy():
    assert is_casadi_type(
        np.array([1, 2, 3]),
        recursive=True
    ) == False
    assert is_casadi_type(
        np.array([1, 2, 3]),
        recursive=False
    ) == False


def test_casadi():
    assert is_casadi_type(
        cas.GenMX_ones(5),
        recursive=False
    ) == True
    assert is_casadi_type(
        cas.GenMX_ones(5),
        recursive=True
    ) == True


def test_numpy_list():
    assert is_casadi_type(
        [np.array(5), np.array(7)],
        recursive=False
    ) == False
    assert is_casadi_type(
        [np.array(5), np.array(7)],
        recursive=True
    ) == False


def test_casadi_list():
    assert is_casadi_type(
        [cas.GenMX_ones(5), cas.GenMX_ones(5)],
        recursive=False
    ) == False
    assert is_casadi_type(
        [cas.GenMX_ones(5), cas.GenMX_ones(5)],
        recursive=True
    ) == True


def test_mixed_list():
    assert is_casadi_type(
        [np.array(5), cas.GenMX_ones(5)],
        recursive=False
    ) == False
    assert is_casadi_type(
        [np.array(5), cas.GenMX_ones(5)],
        recursive=True
    ) == True


def test_multi_level_contaminated_list():
    a = [[1 for _ in range(10)] for _ in range(10)]

    assert is_casadi_type(a, recursive=False) == False
    assert is_casadi_type(a, recursive=True) == False

    a[5][5] = cas.MX(1)

    assert is_casadi_type(a, recursive=False) == False
    assert is_casadi_type(a, recursive=True) == True

    a[5][5] = np.array(cas.DM(1), dtype="O")

    assert is_casadi_type(a, recursive=False) == False
    assert is_casadi_type(a, recursive=True) == False


if __name__ == '__main__':
    pytest.main([__file__])
