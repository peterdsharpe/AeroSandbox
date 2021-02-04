from aerosandbox.optimization.math import *
import pytest


def test_diff():
    a = np.arange(100)

    assert np.all(
        diff(a) == pytest.approx(1)
    )


def test_trapz():
    a = np.arange(100)

    assert diff(trapz(a)) == pytest.approx(1)


def test_invertability_of_diff_trapz():
    a = np.sin(np.arange(10))

    assert np.all(
        trapz(diff(a)) == pytest.approx(diff(trapz(a)))
    )


if __name__ == '__main__':
    pytest.main()
