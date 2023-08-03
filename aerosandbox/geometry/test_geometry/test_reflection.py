from aerosandbox.geometry.common import reflect_over_XZ_plane
import aerosandbox.numpy as np
import pytest
import casadi as cas

vec = np.arange(3)
square = np.arange(9).reshape((3, 3))
rectangular_tall = np.arange(12).reshape((4, 3))
rectangular_wide = np.arange(12).reshape((3, 4))


def test_np_vector():
    assert np.all(
        reflect_over_XZ_plane(vec) ==
        np.array([0, -1, 2])
    )


def test_cas_vector():
    output = reflect_over_XZ_plane(cas.DM(vec))
    assert isinstance(output, cas.DM)
    assert np.all(
        output ==
        np.array([0, -1, 2])
    )


def test_np_vector_2D_wide():
    assert np.all(
        reflect_over_XZ_plane(np.expand_dims(vec, 0)) ==
        np.array([0, -1, 2])
    )


def test_np_vector_2D_tall():
    with pytest.raises(ValueError):
        reflect_over_XZ_plane(np.expand_dims(vec, 1))


def test_np_square():
    assert np.all(
        reflect_over_XZ_plane(square) ==
        np.array([
            [0, -1, 2],
            [3, -4, 5],
            [6, -7, 8]
        ])
    )


def test_cas_square():
    output = reflect_over_XZ_plane(cas.DM(square))
    assert isinstance(output, cas.DM)
    assert np.all(
        output ==
        np.array([
            [0, -1, 2],
            [3, -4, 5],
            [6, -7, 8]
        ])
    )


def test_np_rectangular_tall():
    assert np.all(
        reflect_over_XZ_plane(rectangular_tall) ==
        np.array([
            [0, -1, 2],
            [3, -4, 5],
            [6, -7, 8],
            [9, -10, 11],
        ])
    )


def test_cas_rectangular_tall():
    output = reflect_over_XZ_plane(cas.DM(rectangular_tall))
    assert isinstance(output, cas.DM)
    assert np.all(
        output ==
        np.array([
            [0, -1, 2],
            [3, -4, 5],
            [6, -7, 8],
            [9, -10, 11],
        ])
    )


def test_np_rectangular_wide():
    with pytest.raises(ValueError):
        reflect_over_XZ_plane(rectangular_wide)


def test_cas_rectangular_wide():
    with pytest.raises(ValueError):
        reflect_over_XZ_plane(cas.DM(rectangular_wide))


def test_np_3D():
    with pytest.raises(ValueError):
        reflect_over_XZ_plane(np.arange(2 * 3 * 4).reshape((2, 3, 4)))


if __name__ == '__main__':
    pytest.main()
