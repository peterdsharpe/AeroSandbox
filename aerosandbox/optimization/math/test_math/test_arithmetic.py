from aerosandbox.optimization.math import *
import pytest


def test_sum():
    a = np.arange(101)

    assert sum1(a) == 5050  # Gauss would be proud.


def test_mean():
    a = linspace(0, 10, 50)

    assert mean(a) == pytest.approx(5)


if __name__ == '__main__':
    pytest.main()
