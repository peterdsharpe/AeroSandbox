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


def test_equality():
    p1 = Polygon(
        coordinates=np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ])
    )

    p2 = p1.deepcopy()

    assert p1 == p2

    assert not p1 == p2.translate(0.1, 0)


def test_translate_scale_rotate():
    p1 = Polygon(
        coordinates=np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ])
    )
    p2 = (
        p1
        .translate(1, 1)
        .rotate(np.pi / 2, 1, 1)
        .scale(2, 1)
        .rotate(3 * np.pi / 2, 2, 1)
        .scale(1, 0.5)
        .translate(-2, -0.5)
    )

    assert np.allclose(
        p1.coordinates,
        p2.coordinates
    )


def test_jaccard_similarity():
    try:
        import shapely
    except ImportError:
        print("Shapely (optional) not installed; skipping test_jaccard_similarity.")
        return

    p1 = Polygon(
        coordinates=np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ])
    )
    p2 = p1.copy()

    assert p1.jaccard_similarity(p2) == pytest.approx(1)
    assert p1.jaccard_similarity(p2.translate(0.5, 0)) == pytest.approx(1 / 3)
    assert p1.jaccard_similarity(p2.translate(1, 0)) == pytest.approx(0)
    assert p1.jaccard_similarity(p2.translate(1.5, 0)) == pytest.approx(0)
    assert p1.jaccard_similarity(p2.translate(0.5, 0.5)) == pytest.approx(1 / 7)
    assert p1.jaccard_similarity(p2.translate(1, 1)) == pytest.approx(0)

    assert p1.jaccard_similarity(p2.rotate(np.pi / 2, 0.5, 0.5)) == pytest.approx(1)


if __name__ == '__main__':
    # test_translate_scale_rotate()
    pytest.main()
