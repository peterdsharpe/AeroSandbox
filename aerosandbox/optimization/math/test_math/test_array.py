from aerosandbox.optimization.math import array
import pytest
import numpy as np
import casadi as cas


def test_numpy_equivalency_1D():
    inputs = [
        1,
        2
    ]

    a = array(inputs)
    a_np = np.array(inputs)

    assert np.all(a == a_np)


def test_numpy_equivalency_2D():
    inputs = [
        [1, 2],
        [3, 4]
    ]

    a = array(inputs)
    a_np = np.array(inputs)

    assert np.all(a == a_np)

def test_casadi_1D_shape():
    a = array([cas.DM(1), cas.DM(2)])

if __name__ == '__main__':
    pytest.main()
