from aerosandbox.numpy import linalg
from aerosandbox.numpy.array import array
import numpy as _onp


def rotation_matrix_2D(
        angle,
):
    """
    Gives the 2D rotation matrix associated with a counterclockwise rotation about an angle.
    Args:
        angle: Angle by which to rotate. Given in radians.

    Returns: The 2D rotation matrix

    """
    sintheta = _onp.sin(angle)
    costheta = _onp.cos(angle)
    rotation_matrix = array([
        [costheta, -sintheta],
        [sintheta, costheta]
    ])
    return rotation_matrix


def rotation_matrix_3D(
        angle,
        axis,
        _axis_already_normalized=False
):
    """
    Gives the 3D rotation matrix from an angle and an axis.
    An implmentation of https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    :param angle: can be one angle or a vector (1d ndarray) of angles. Given in radians. # TODO note deprecated functionality; must be scalar
        Direction corresponds to the right-hand rule.
    :param axis: a 1d numpy array of length 3 (x,y,z). Represents the angle.
    :param _axis_already_normalized: boolean, skips normalization for speed if you flag this true.
    :return:
        * If angle is a scalar, returns a 3x3 rotation matrix.
        * If angle is a vector, returns a 3x3xN rotation matrix.
    """
    if not _axis_already_normalized:
        axis = axis / linalg.norm(axis)

    sintheta = _onp.sin(angle)
    costheta = _onp.cos(angle)
    cpm = array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])  # The cross product matrix of the rotation axis vector
    outer_axis = linalg.outer(axis, axis)

    rot_matrix = costheta * _onp.eye(3) + sintheta * cpm + (1 - costheta) * outer_axis
    return rot_matrix
