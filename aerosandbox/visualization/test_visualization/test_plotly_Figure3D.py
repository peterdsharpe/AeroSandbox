import numpy as onp
import pytest

pytest.importorskip("plotly")

from aerosandbox.visualization.plotly_Figure3D import (  # noqa: E402
    Figure3D,
    reflect_over_XZ_plane,
)


def test_reflect_over_XZ_plane_accepts_tuples_and_lists():
    assert onp.all(
        reflect_over_XZ_plane((1.0, 2.0, 3.0)) == onp.array([1.0, -2.0, 3.0])
    )
    assert onp.all(
        reflect_over_XZ_plane([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        == onp.array([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]])
    )


def test_reflect_over_XZ_plane_rejects_bad_shapes():
    with pytest.raises(ValueError):
        reflect_over_XZ_plane([(1.0, 2.0), (3.0, 4.0)])


def test_add_line_mirror_with_tuple_points():
    """
    Regression test: mirror=True used to crash with AttributeError on
    tuple/list points, which is the exact format shown in the docstrings.
    """
    f = Figure3D()
    f.add_line([(0, 0, 0), (1, 1, 0)], mirror=True)
    assert f.y_line == [0.0, 1.0, None, 0.0, -1.0, None]


def test_add_streamline_mirror_with_tuple_points():
    f = Figure3D()
    f.add_streamline([(0, 2, 0), (1, 3, 0)], mirror=True)
    assert f.y_streamline == [2.0, 3.0, None, -2.0, -3.0, None]


def test_add_tri_mirror_with_tuple_points():
    f = Figure3D()
    f.add_tri([(0, 0, 0), (1, 0, 0), (0, 1, 0)], mirror=True)
    assert f.y_face == [0.0, 0.0, 1.0, 0.0, 0.0, -1.0]


def test_add_quad_mirror_with_tuple_points():
    f = Figure3D()
    f.add_quad([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], mirror=True)
    assert f.y_face == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0, -1.0]


if __name__ == "__main__":
    pytest.main([__file__])
