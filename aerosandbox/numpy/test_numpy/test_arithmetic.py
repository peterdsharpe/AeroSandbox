import aerosandbox.numpy as np
import casadi as cas
import pytest


def test_sum():
    a = np.arange(101)

    assert np.sum(a) == 5050  # Gauss would be proud.


def test_sum2():
    # Check it returns the same results with casadi and numpy
    a = np.array([[1, 2, 3], [1, 2, 3]])
    b = cas.SX(a)

    assert np.all(np.sum(a) == cas.DM(np.sum(b)))
    assert np.all(np.sum(a, axis=1) == cas.DM(np.sum(b, axis=1)))


def test_mean():
    a = np.linspace(0, 10, 50)

    assert np.mean(a) == pytest.approx(5)


def test_mean_2D_casadi_axis_None_returns_scalar():
    """np.mean(axis=None) on a 2D CasADi matrix should return a scalar
    (regression test: it used to return a column vector)."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    cas_a = cas.DM(a)

    result = np.mean(cas_a)
    assert np.prod(result.shape) == 1  # Scalar (CasADi scalars are 1x1)
    assert float(result) == pytest.approx(np.mean(a))

    # axis=0 / axis=1 behavior is unchanged:
    assert np.all(cas.DM(np.mean(cas_a, axis=0)) == np.mean(a, axis=0))
    assert np.all(cas.DM(np.mean(cas_a, axis=1)) == np.mean(a, axis=1))


def test_cumsum():
    n = np.arange(6).reshape((3, 2))

    assert np.all(np.cumsum(n) == np.array([0, 1, 3, 6, 10, 15]))
    # assert np.all( # TODO add casadi testing here
    #     np.cumsum(c) == np.array([0, 1, 3, 6, 10, 15])
    # )


if __name__ == "__main__":
    pytest.main()
