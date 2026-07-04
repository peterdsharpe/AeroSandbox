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


def test_mod_matches_numpy_for_casadi_types():
    """np.mod() should agree with NumPy for CasADi inputs, including negative
    dividends/divisors and exact multiples (regression test: mod(DM(-4), 2)
    used to return 2 instead of 0)."""
    dividends = np.arange(-4, 4.5, 0.5)
    for divisor in [2, -2, 3.7, -3.7, 0.5]:
        expected = np.mod(dividends, divisor)

        # Scalar CasADi inputs
        for dividend, expect in zip(dividends, expected):
            assert float(np.mod(cas.DM(dividend), divisor)) == pytest.approx(
                expect
            ), f"mod({dividend}, {divisor})"

        # Vector CasADi inputs
        result = cas.DM(np.mod(cas.DM(dividends), divisor)).full().flatten()
        assert result == pytest.approx(expected)


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


def test_round_dual_backend():
    """np.round() should support CasADi types (regression test: it used to be
    NumPy's round, which raises TypeError on cas.MX)."""
    # NumPy backend: identical to numpy.round
    a = np.array([1.4, 1.6, -1.4, -1.6])
    assert np.all(np.round(a) == np.array([1.0, 2.0, -1.0, -2.0]))
    assert np.round(1.2345, 2) == pytest.approx(1.23)

    # CasADi numeric backend
    assert float(np.round(cas.DM(1.6))) == pytest.approx(2.0)
    assert float(np.round(cas.DM(-1.4))) == pytest.approx(-1.0)
    assert float(np.round(cas.DM(1.2345), decimals=2)) == pytest.approx(1.23)

    # CasADi symbolic backend
    x = cas.MX.sym("x")
    f = cas.Function("f", [x], [np.round(x)])
    assert float(f(3.7)) == pytest.approx(4.0)
    assert float(f(-2.2)) == pytest.approx(-2.0)


def test_cumsum():
    n = np.arange(6).reshape((3, 2))

    assert np.all(np.cumsum(n) == np.array([0, 1, 3, 6, 10, 15]))
    # assert np.all( # TODO add casadi testing here
    #     np.cumsum(c) == np.array([0, 1, 3, 6, 10, 15])
    # )


if __name__ == "__main__":
    pytest.main()
