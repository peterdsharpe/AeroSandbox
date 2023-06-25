import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.geometry.polygon import Polygon
import pytest


def test_polygon_creation():
    p = Polygon(
        coordinates=np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ])
    )
    assert p.n_points() == 4
    assert np.all(p.x() == np.array([0, 1, 1, 0]))
    assert np.all(p.y() == np.array([0, 0, 1, 1]))


def test_contains_points():
    p = Polygon(
        coordinates=np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ])
    )

    assert p.contains_points(0.5, 0.5) == True
    assert p.contains_points(-0.1, 0.5) == False
    assert p.contains_points(0.5, -0.1) == False
    assert p.contains_points(-0.1, -0.1) == False
    assert p.contains_points(1.1, 1.1) == False
    assert p.contains_points(1.0, 1.0) == True
    assert p.contains_points(0.5, 1.0) == True
    assert p.contains_points(0.5, 1.1) == False

    assert np.all(p.contains_points(
        x=np.array([0.5, 0.5, -0.1, -0.1]),
        y=np.array([0.5, -0.1, 0.5, -0.1])
    ) == np.array([True, False, False, False]))

    shape = (1, 2, 3, 4)
    x_points = np.random.randn(*shape)
    y_points = np.random.randn(*shape)
    contains = p.contains_points(x_points, y_points)
    assert shape == contains.shape


if __name__ == '__main__':
    pytest.main()
