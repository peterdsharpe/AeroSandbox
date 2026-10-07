"""Rotation matrix utilities for the AeroSandbox NumPy-like interface.

This module provides functions for generating rotation matrices in 2D and 3D,
working with both NumPy arrays and CasADi symbolic arrays.
"""

from typing import Literal, overload
from aerosandbox.numpy import sin, cos, linalg
from aerosandbox.numpy.array import array
from aerosandbox.numpy.typing import Vectorizable, ArrayLike, Array
import numpy as _onp


@overload
def rotation_matrix_2D(
    angle: Vectorizable,
    as_array: Literal[True] = True,
) -> Array: ...


@overload
def rotation_matrix_2D(
    angle: Vectorizable,
    as_array: Literal[False],
) -> list[list[Vectorizable]]: ...


def rotation_matrix_2D(
    angle: Vectorizable,
    as_array: bool = True,
) -> Array | list[list[Vectorizable]]:
    """Give the 2D rotation matrix for a counterclockwise rotation.

    Parameters
    ----------
    angle : Vectorizable
        Angle by which to rotate, in radians.
    as_array : bool, optional
        If True (default), return an array. If False, return a list of lists.

    Returns
    -------
    Array | list[list[Vectorizable]]
        The 2x2 rotation matrix. Elements are Vectorizable when angle is.
    """
    s = sin(angle)
    c = cos(angle)
    rot: list[list[Vectorizable]] = [[c, -s], [s, c]]
    if as_array:
        return array(rot)
    else:
        return rot


@overload
def rotation_matrix_3D(
    angle: Vectorizable,
    axis: ArrayLike | str,
    as_array: Literal[True] = True,
    axis_already_normalized: bool = False,
) -> Array: ...


@overload
def rotation_matrix_3D(
    angle: Vectorizable,
    axis: ArrayLike | str,
    as_array: Literal[False],
    axis_already_normalized: bool = False,
) -> list[list[Vectorizable]]: ...


def rotation_matrix_3D(
    angle: Vectorizable,
    axis: ArrayLike | str,
    as_array: bool = True,
    axis_already_normalized: bool = False,
) -> Array | list[list[Vectorizable]]:
    """Give the 3D rotation matrix for a rotation about a given axis.

    An implementation of the axis-angle rotation matrix formula.

    Parameters
    ----------
    angle : Vectorizable
        The angle to rotate by, in radians. Direction of rotation corresponds
        to the right-hand rule. Can be vectorized.
    axis : ArrayLike | str
        The axis to rotate about. Can be:

        - A string: 'x', 'y', or 'z' for principal axes
        - An array-like of shape (3,) or (3, N) for arbitrary axes

        Can be vectorized; if so, axis[0] yields all the x-components, etc.
    as_array : bool, optional
        If True (default), return an array. If False, return a list of lists.
        If vectorizing, it is recommended to set this to False.
    axis_already_normalized : bool, optional
        If True, skip axis normalization for speed. Default is False.

    Returns
    -------
    Array | list[list[Vectorizable]]
        The 3x3 rotation matrix. Elements are Vectorizable when angle is.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    """
    s = sin(angle)
    c = cos(angle)

    if isinstance(axis, str):
        if axis.lower() == "x":
            rot = [[1, 0, 0], [0, c, -s], [0, s, c]]
        elif axis.lower() == "y":
            rot = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        elif axis.lower() == "z":
            rot = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        else:
            raise ValueError("If `axis` is a string, it must be `x`, `y`, or `z`.")
    else:
        ux = axis[0]
        uy = axis[1]
        uz = axis[2]

        if not axis_already_normalized:
            norm = (ux**2 + uy**2 + uz**2) ** 0.5
            ux = ux / norm
            uy = uy / norm
            uz = uz / norm

        rot = [
            [
                c + ux**2 * (1 - c),
                ux * uy * (1 - c) - uz * s,
                ux * uz * (1 - c) + uy * s,
            ],
            [
                uy * ux * (1 - c) + uz * s,
                c + uy**2 * (1 - c),
                uy * uz * (1 - c) - ux * s,
            ],
            [
                uz * ux * (1 - c) - uy * s,
                uz * uy * (1 - c) + ux * s,
                c + uz**2 * (1 - c),
            ],
        ]

    if as_array:
        return array(rot)
    else:
        return rot


@overload
def rotation_matrix_from_euler_angles(
    roll_angle: Vectorizable = 0,
    pitch_angle: Vectorizable = 0,
    yaw_angle: Vectorizable = 0,
    as_array: Literal[True] = True,
) -> Array: ...


@overload
def rotation_matrix_from_euler_angles(
    roll_angle: Vectorizable = 0,
    pitch_angle: Vectorizable = 0,
    yaw_angle: Vectorizable = 0,
    as_array: Literal[False] = ...,
) -> list[list[Vectorizable]]: ...


def rotation_matrix_from_euler_angles(
    roll_angle: Vectorizable = 0,
    pitch_angle: Vectorizable = 0,
    yaw_angle: Vectorizable = 0,
    as_array: bool = True,
) -> Array | list[list[Vectorizable]]:
    """Give the rotation matrix corresponding to Euler angle rotations.

    Uses the standard (yaw, pitch, roll) Euler angle convention where:

    1. First, a rotation about x is applied (roll)
    2. Second, a rotation about y is applied (pitch)
    3. Third, a rotation about z is applied (yaw)

    In matrix form: R = R_z(yaw) @ R_y(pitch) @ R_x(roll).

    To transform from body axes to earth axes, pre-multiply your vector::

        vector_earth = rotation_matrix_from_euler_angles(...) @ vector_body

    Parameters
    ----------
    roll_angle : Vectorizable, optional
        Rotation about the x-axis, in radians. Default is 0.
    pitch_angle : Vectorizable, optional
        Rotation about the y-axis, in radians. Default is 0.
    yaw_angle : Vectorizable, optional
        Rotation about the z-axis, in radians. Default is 0.
    as_array : bool, optional
        If True (default), return an array. If False, return a list of lists.

    Returns
    -------
    Array | list[list[Vectorizable]]
        The 3x3 rotation matrix. Elements are Vectorizable when angles are.

    References
    ----------
    .. [1] http://planning.cs.uiuc.edu/node102.html
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
        [-sb, cb * sc, cb * cc],
    ]

    if as_array:
        return array(rot)
    else:
        return rot


def is_valid_rotation_matrix(a: Array, tol: float = 1e-9) -> bool:
    """Check whether a matrix satisfies the properties of a rotation matrix.

    Tests for:

    - Volume-preserving (determinant = 1)
    - Right-handedness of output reference frame
    - Orthogonality of output reference frame

    Parameters
    ----------
    a : Array
        The matrix to test.
    tol : float, optional
        Tolerance for floating-point comparisons. Default is 1e-9.

    Returns
    -------
    bool
        True if the matrix is a valid rotation matrix, False otherwise.
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

    return is_volume_preserving_and_right_handed and is_orthogonality_preserving
