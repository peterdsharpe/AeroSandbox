import aerosandbox.numpy as np
import pytest


def test_diff():
    a = np.arange(100)

    assert np.all(
        np.diff(a) == pytest.approx(1)
    )


def test_trapz():
    a = np.arange(100)

    assert np.diff(np.trapz(a)) == pytest.approx(1)


def test_invertability_of_diff_trapz():
    a = np.sin(np.arange(10))

    assert np.all(
        np.trapz(np.diff(a)) == pytest.approx(np.diff(np.trapz(a)))
    )


if __name__ == '__main__':
    pytest.main()
