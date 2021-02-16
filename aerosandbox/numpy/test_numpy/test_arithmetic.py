import aerosandbox.numpy as np
import casadi as cas
import pytest


def test_sum():
    a = np.arange(101)

    assert np.sum(a) == 5050  # Gauss would be proud.
    
def test_sum2():
    # Check it returns the same results with casadi and numpy
    a = np.array([[1,2,3],[1,2,3]])
    b = cas.SX(a)
    
    assert np.all(np.sum(a) == cas.DM(np.sum(b)))
    assert np.all(np.sum(a, axis=1) == cas.DM(np.sum(b, axis=1)))


def test_mean():
    a = np.linspace(0, 10, 50)

    assert np.mean(a) == pytest.approx(5)


def test_cumsum():
    a = np.array([1, 2, 3])
    
    assert np.all(np.cumsum(a) == np.array([ 1,  3,  6]))
    

if __name__ == '__main__':
    pytest.main()
