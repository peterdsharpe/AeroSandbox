import aerosandbox.numpy as np
import pytest
import casadi as cas


def test_interp():
    x_np = np.arange(5)
    y_np = x_np + 10

    for x, y in zip(
            [x_np, x_np],
            [y_np, cas.DM(y_np)]
    ):
        assert np.interp(0, x, y) == pytest.approx(10)
        assert np.interp(4, x, y) == pytest.approx(14)
        assert np.interp(0.5, x, y) == pytest.approx(10.5)
        assert np.interp(-1, x, y) == pytest.approx(10)
        assert np.interp(5, x, y) == pytest.approx(14)
        assert np.interp(-1, x, y, left=-10) == pytest.approx(-10)
        assert np.interp(5, x, y, right=-10) == pytest.approx(-10)
        assert np.interp(5, x, y, period=4) == pytest.approx(11)


if __name__ == '__main__':
    test_interp()
