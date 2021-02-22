import aerosandbox.numpy as np
import pytest


def test_basic_logicals_numpy():
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])

    assert np.all(
        a & b == np.array([True, False, False, False])
    )


if __name__ == '__main__':
    pytest.main()
