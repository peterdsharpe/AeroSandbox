import pytest
import aerosandbox.numpy as np


def test_euler_angles_equivalence_to_general_3D():
    phi = 1
    theta = 2
    psi = 3

    rot_euler = np.rotation_matrix_from_euler_angles(phi, theta, psi)
    rot_manual = (
            np.rotation_matrix_3D(psi, np.array([0, 0, 1])) @
            np.rotation_matrix_3D(theta, np.array([0, 1, 0])) @
            np.rotation_matrix_3D(phi, np.array([1, 0, 0]))
    )

    assert rot_euler == pytest.approx(rot_manual)


def test_validity_of_euler_angles():
    rot = np.rotation_matrix_from_euler_angles(
        2, 4, 6
    )
    assert np.is_valid_rotation_matrix(rot)


def test_validity_of_general_3D():
    rot = np.rotation_matrix_3D(
        angle=1,
        axis=[2, 3, 4]
    )
    assert np.is_valid_rotation_matrix(rot)


def test_validity_checker():
    rot = np.eye(3)
    assert np.is_valid_rotation_matrix(rot)

    rot[2, 2] = -1
    assert not np.is_valid_rotation_matrix(rot)

    rot[2, 2] = 1 + 1e-6
    assert not np.is_valid_rotation_matrix(rot)

    rot[2, 2] = 1
    assert np.is_valid_rotation_matrix(rot)


def test_general_3D_shorthands():
    rotx = np.rotation_matrix_3D(1, np.array([1, 0, 0]))
    assert pytest.approx(rotx) == np.rotation_matrix_3D(1, "x")

    roty = np.rotation_matrix_3D(1, np.array([0, 1, 0]))
    assert pytest.approx(roty) == np.rotation_matrix_3D(1, "y")

    rotz = np.rotation_matrix_3D(1, np.array([0, 0, 1]))
    assert pytest.approx(rotz) == np.rotation_matrix_3D(1, "z")


if __name__ == '__main__':
    pytest.main()
