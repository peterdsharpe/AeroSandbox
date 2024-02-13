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
    Yields the rotation matrix that corresponds to a rotation by a specified amount about a given axis.

    An implementation of https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle

    Args:

        angle: The angle to rotate by. [radians]
        Direction of rotation corresponds to the right-hand rule.
        Can be vectorized.

        axis: The axis to rotate about. [ndarray]
        Can be vectorized; be sure axis[0] yields all the x-components, etc.

        as_array: boolean, returns a 3x3 array-like if True, and a list-of-lists otherwise.

            If you are intending to use this function vectorized, it is recommended you flag this False. (Or test before
            proceeding.)

        axis_already_normalized: boolean, skips axis normalization for speed if you flag this true.

    Returns:
        The rotation matrix, with type according to the parameter `as_array`.
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
        ux = axis[0]
        uy = axis[1]
        uz = axis[2]

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

    See notes:
    http://planning.cs.uiuc.edu/node102.html

    Args:
        roll_angle: The roll angle, which is a rotation about the x-axis. [radians]
        pitch_angle: The pitch angle, which is a rotation about the y-axis. [radians]
        yaw_angle: The yaw angle, which is a rotation about the z-axis. [radians]
        as_array: If True, returns a 3x3 array-like. If False, returns a list-of-lists.

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
) -> bool:
    """
    Returns a boolean of whether the given matrix satisfies the properties of a rotation matrix.

    Specifically, tests for:
        * Volume-preserving
        * Handedness of output reference frame
        * Orthogonality of output reference frame

    Args:
        a: The array-like to be tested
        tol: A tolerance to use for truthiness; accounts for floating-point error.

    Returns: A boolean of whether the array-like is a valid rotation matrix.

    """

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
