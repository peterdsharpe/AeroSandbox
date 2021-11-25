import pytest
import aerosandbox.numpy as np
import casadi as cas


def test_cross_1D_input():
    a = np.array([1, 1, 1])
    b = np.array([1, 2, 3])

    cas_a = cas.DM(a)
    cas_b = cas.DM(b)

    correct_result = np.cross(a, b)
    cas_correct_result = cas.DM(correct_result)

    assert np.all(
        np.cross(a, cas_b) == cas_correct_result
    )
    assert np.all(
        np.cross(cas_a, b) == cas_correct_result
    )
    assert np.all(
        np.cross(cas_a, cas_b) == cas_correct_result
    )


def test_cross_2D_input_last_axis():
    a = np.tile(np.array([1, 1, 1]), (3, 1))
    b = np.tile(np.array([1, 2, 3]), (3, 1))

    cas_a = cas.DM(a)
    cas_b = cas.DM(b)

    correct_result = np.cross(a, b)
    cas_correct_result = cas.DM(correct_result)

    assert np.all(
        np.cross(a, cas_b) == cas_correct_result
    )
    assert np.all(
        np.cross(cas_a, b) == cas_correct_result
    )
    assert np.all(
        np.cross(cas_a, cas_b) == cas_correct_result
    )


def test_cross_2D_input_first_axis():
    a = np.tile(np.array([1, 1, 1]), (3, 1)).T
    b = np.tile(np.array([1, 2, 3]), (3, 1)).T

    cas_a = cas.DM(a)
    cas_b = cas.DM(b)

    correct_result = np.cross(a, b, axis=0)
    cas_correct_result = cas.DM(correct_result)

    assert np.all(
        np.cross(a, cas_b, axis=0) == cas_correct_result
    )
    assert np.all(
        np.cross(cas_a, b, axis=0) == cas_correct_result
    )
    assert np.all(
        np.cross(cas_a, cas_b, axis=0) == cas_correct_result
    )


if __name__ == '__main__':
    pytest.main()
