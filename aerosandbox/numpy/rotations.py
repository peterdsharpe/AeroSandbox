from aerosandbox.numpy import sin, cos, linalg
from aerosandbox.numpy.array import array
import numpy as _onp
from typing import Union, List


def rotation_matrix_2D(
        angle,
        as_array: bool = True,
):
    """
    Gives the 2D rotation matrix associated with a counterclockwise rotation about an angle.
    Args:
        angle: Angle by which to rotate. Given in radians.
        as_array: Determines whether to return an array-like or just a simple list of lists.

    Returns: The 2D rotation matrix

    """
    s = sin(angle)
    c = cos(angle)
    rot = [
        [c, -s],
        [s, c]
    ]
    if as_array:
        return array(rot)
    else:
        return rot


def rotation_matrix_3D(
        angle: Union[float, _onp.ndarray],
        axis: Union[_onp.ndarray, List, str],
        as_array: bool = True,
        axis_already_normalized: bool = False
):
    """
    Gives the 3D rotation matrix from an angle and an axis.
    An implementation of https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    :param angle: can be one angle or a vector (1d ndarray) of angles. Given in radians. # TODO note deprecated functionality; must be scalar
        Direction corresponds to the right-hand rule.
    :param axis: a 1d numpy array of length 3 (x,y,z). Represents the angle.
    :param axis_already_normalized: boolean, skips normalization for speed if you flag this true.
    :return:
        * If angle is a scalar, returns a 3x3 rotation matrix.
        * If angle is a vector, returns a 3x3xN rotation matrix.
    """
    s = sin(angle)
    c = cos(angle)

    if isinstance(axis, str):
        if axis.lower() == "x":
            rot = [
                [1, 0, 0],
                [0, c, -s],
                [0, s, c]
            ]
        elif axis.lower() == "y":
            rot = [
                [c, 0, s],
                [0, 1, 0],
                [-s, 0, c]
            ]
        elif axis.lower() == "z":
            rot = [
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ]
        else:
            raise ValueError("If `axis` is a string, it must be `x`, `y`, or `z`.")
    else:
        ux, uy, uz = axis

        if not axis_already_normalized:
            norm = (ux ** 2 + uy ** 2 + uz ** 2) ** 0.5
            ux = ux / norm
            uy = uy / norm
            uz = uz / norm

        rot = [
            [c + ux ** 2 * (1 - c), ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s],
            [uy * ux * (1 - c) + uz * s, c + uy ** 2 * (1 - c), uy * uz * (1 - c) - ux * s],
            [uz * ux * (1 - c) - uy * s, uz * uy * (1 - c) + ux * s, c + uz ** 2 * (1 - c)]
        ]

    if as_array:
        return array(rot)
    else:
        return rot


def rotation_matrix_from_euler_angles(
        roll_angle: Union[float, _onp.ndarray] = 0,
        pitch_angle: Union[float, _onp.ndarray] = 0,
        yaw_angle: Union[float, _onp.ndarray] = 0,
        as_array: bool = True
):
    """
    Yields the rotation matrix that corresponds to a given Euler angle rotation.

    Note: This uses the standard (yaw, pitch, roll) Euler angle rotation, where:
    * First, a rotation about x is applied (roll)
    * Second, a rotation about y is applied (pitch)
    * Third, a rotation about z is applied (yaw)

    In other words: R = R_z(yaw) @ R_y(pitch) @ R_x(roll).

    Note: To use this, pre-multiply your vector to go from body axes to earth axes.
        Example:
            >>> vector_earth = rotation_matrix_from_euler_angles(np.pi / 4, np.pi / 4, np.pi / 4) @ vector_body

    Args:
        roll_angle: The roll angle, which is a rotation about the x-axis. [radians]
        pitch_angle: The pitch angle, which is a rotation about the y-axis. [radians]
        yaw_angle: The yaw angle, which is a rotation about the z-axis. [radians]
        as_array:

    Returns:

    """
    sa = sin(yaw_angle)
    ca = cos(yaw_angle)
    sb = sin(pitch_angle)
    cb = cos(pitch_angle)
    sc = sin(roll_angle)
    cc = cos(roll_angle)

    rot = [
        [ca * cb, ca * sb * sc - sa * cc, ca * sb * cc + sa * sc],
        [sa * cb, sa * sb * sc + ca * cc, sa * sb * cc - ca * sc],
        [-sb, cb * sc, cb * cc]
    ]

    if as_array:
        return array(rot)
    else:
        return rot


def is_valid_rotation_matrix(
        a: _onp.ndarray,
        tol=1e-9
):
    def approx_equal(x, y):
        return (x > y - tol) and (x < y + tol)

    det = linalg.det(a)
    is_volume_preserving_and_right_handed = approx_equal(det, 1)

    eye_approx = a.T @ a
    eye = _onp.eye(a.shape[0])
    is_orthogonality_preserving = True
    for i in range(eye.shape[0]):
        for j in range(eye.shape[1]):
            if not approx_equal(eye_approx[i, j], eye[i, j]):
                is_orthogonality_preserving = False

    return (
            is_volume_preserving_and_right_handed and
            is_orthogonality_preserving
    )
