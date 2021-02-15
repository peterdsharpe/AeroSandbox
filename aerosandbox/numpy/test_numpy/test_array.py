from aerosandbox.numpy.array import *
import pytest
import aerosandbox.numpy as np
import casadi as cas
# import aerosandbox as asb


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


def test_length():
    assert length(5) == 1
    assert length(5.) == 1
    assert length([1, 2, 3]) == 3

    assert length(np.array(5)) == 1
    assert length(np.array([5])) == 1
    assert length(np.array([1, 2, 3])) == 3
    assert length(np.ones((3, 2))) == 3

    assert length(cas.GenMX_ones(5)) == 5


def test_roll_onp():
    a = [1, 2, 3]
    b = [3, 1, 2]
    
    assert np.roll(a, 1) == b
    
    
def test_roll_casadi():
    a = cas.SX([1,2,3])
    b = np.array([3,1,2])
      
    assert all(np.array(cas.DM(np.roll(a, 1)) == b))
    
    
def test_max():
    a = cas.SX([1,2,3])
    b = [1,2,3]
    
    assert int(np.max(a)) == int(np.max(b))
    

def test_min():
    a = cas.SX([1,2,3])
    b = [1,2,3]
    
    assert int(np.min(a)) == int(np.min(b))


if __name__ == '__main__':
    pytest.main()
