import aerosandbox.numpy as np
import casadi as cas
import pytest


def test_diff():
    a = np.arange(100)

    assert np.all(np.diff(a) == pytest.approx(1))


def test_diff_casadi_higher_order():
    """np.diff() with n >= 2 on CasADi types should difference iteratively
    (regression test: it used to return the first difference for any n)."""
    a = np.arange(5, dtype=float) ** 2  # [0, 1, 4, 9, 16]

    for n in [1, 2, 3]:
        expected = np.diff(a, n=n)
        result = cas.DM(np.diff(cas.DM(a), n=n)).full().flatten()
        assert len(result) == len(expected)
        assert result == pytest.approx(expected)


def test_gradient_no_varargs_casadi():
    # Zero varargs should default to unit spacing, even for CasADi arrays
    # (which are always 2D, so this exercises the varargs-expansion logic).
    x = cas.DM([1.0, 2.0, 4.0, 7.0])

    grad = np.gradient(x)

    assert grad.full().flatten() == pytest.approx(np.gradient([1.0, 2.0, 4.0, 7.0]))


def test_gradient_no_varargs_2D_higher_order():
    f = np.arange(25, dtype=float).reshape(5, 5) ** 2

    grads = np.gradient(f, n=2)

    assert len(grads) == 2
    for grad in grads:
        assert grad.shape == f.shape


def test_trapz():
    a = np.arange(100)

    assert np.diff(np.trapz(a)) == pytest.approx(1)


def test_invertability_of_diff_trapz():
    a = np.sin(np.arange(10))

    assert np.all(np.trapz(np.diff(a)) == pytest.approx(np.diff(np.trapz(a))))


if __name__ == "__main__":
    pytest.main()
