from aerosandbox.numpy.array import *
import pytest
import aerosandbox.numpy as np
import casadi as cas


def test_array_numpy_equivalency_1D():
    inputs = [
        1,
        2
    ]

    a = array(inputs)
    a_np = np.array(inputs)

    assert np.all(a == a_np)


def test_array_numpy_equivalency_2D():
    inputs = [
        [1, 2],
        [3, 4]
    ]

    a = array(inputs)
    a_np = np.array(inputs)

    assert np.all(a == a_np)


def test_array_casadi_1D_shape():
    a = array([cas.DM(1), cas.DM(2)])
    assert length(a) == 2


def test_can_convert_DM_to_ndarray():
    c = cas.DM([1, 2, 3])
    n = np.array(c)

    assert np.all(n == np.array([1, 2, 3]))


def test_length():
    assert length(5) == 1
    assert length(5.) == 1
    assert length([1, 2, 3]) == 3

    assert length(np.array(5)) == 1
    assert length(np.array([5])) == 1
    assert length(np.array([1, 2, 3])) == 3
    assert length(np.ones((3, 2))) == 3

    assert length(cas.GenMX_ones(5)) == 5


def test_concatenate():
    n = np.arange(10)
    c = cas.DM(n)

    assert concatenate((n, n)).shape == (20,)
    assert concatenate((n, c)).shape == (20, 1)
    assert concatenate((c, n)).shape == (20, 1)
    assert concatenate((c, c)).shape == (20, 1)

    assert concatenate((n, n, n)).shape == (30,)
    assert concatenate((c, c, c)).shape == (30, 1)


def test_stack():
    n = np.arange(10)
    c = cas.DM(n)

    assert stack((n, n)).shape == (2, 10)
    assert stack((n, n), axis=-1).shape == (10, 2)
    assert stack((n, c)).shape == (2, 10)
    assert stack((n, c), axis=-1).shape == (10, 2)
    assert stack((c, c)).shape == (2, 10)
    assert stack((c, c), axis=-1).shape == (10, 2)

    with pytest.raises(Exception):
        assert stack((n, n), axis=2)

    with pytest.raises(Exception):
        stack((c, c), axis=2)


def test_roll_onp():
    a = [1, 2, 3]
    b = [3, 1, 2]

    assert np.all(np.roll(a, 1) == b)


def test_roll_casadi():
    b = np.array([[3, 1, 2]])
    a = cas.SX(b)

    assert np.all(cas.DM(np.roll(a, 1)) == b)


def test_roll_casadi_2d():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = cas.SX(a)

    assert np.all(cas.DM(np.roll(b, 1, axis=1)) == np.roll(a, 1, axis=1))


def test_max():
    a = cas.SX([1, 2, 3])
    b = [1, 2, 3]

    assert int(np.max(a)) == int(np.max(b))


def test_min():
    a = cas.SX([1, 2, 3])
    b = [1, 2, 3]

    assert int(np.min(a)) == int(np.min(b))


if __name__ == '__main__':
    test_can_convert_DM_to_ndarray()
    pytest.main()
