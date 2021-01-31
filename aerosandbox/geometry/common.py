import copy
from aerosandbox.tools.string_formatting import eng_string
from aerosandbox.tools.casadi_functions import *
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns


def reflect_over_XZ_plane(input_vector):
    """
    Takes in a vector or an array and flips the y-coordinates.
    :param input_vector: A vector or list of vectors to flip.
    :return: Vector with flipped sign on y-coordinate.
    """
    output_vector = input_vector
    shape = output_vector.shape
    if len(shape) == 1 and shape[0] == 3:
        output_vector = output_vector * cas.vertcat(1, -1, 1)
    elif len(shape) == 2 and shape[1] == 1 and shape[0] == 3:  # Vector of 3 items
        output_vector = output_vector * cas.vertcat(1, -1, 1)
    elif len(shape) == 2 and shape[1] == 3:  # 2D Nx3 vector
        output_vector = cas.horzcat(output_vector[:, 0], -1 * output_vector[:, 1], output_vector[:, 2])
    # elif len(shape) == 3 and shape[2] == 3:  # 3D MxNx3 vector
    #     output_vector = output_vector * cas.array([1, -1, 1])
    else:
        raise Exception("Invalid input for reflect_over_XZ_plane!")

    return output_vector


def rotation_matrix_2D(
        angle,
):
    """
    Gives the 2D rotation matrix associated with a counterclockwise rotation about an angle.
    Args:
        angle: Angle by which to rotate. Given in radians.

    Returns: The 2D rotation matrix

    """
    sintheta = np.sin(angle)
    costheta = np.cos(angle)
    rotation_matrix = np.array([
        [costheta, -sintheta],
        [sintheta, costheta]
    ])
    return rotation_matrix


def rotation_matrix_angle_axis(
        angle,
        axis,
        _axis_already_normalized=False
):
    """
    Gives the 3D rotation matrix from an angle and an axis.
    An implmentation of https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    :param angle: can be one angle or a vector (1d ndarray) of angles. Given in radians.
        Direction corresponds to the right-hand rule.
    :param axis: a 1d numpy array of length 3 (x,y,z). Represents the angle.
    :param _axis_already_normalized: boolean, skips normalization for speed if you flag this true.
    :return:
        * If angle is a scalar, returns a 3x3 rotation matrix.
        * If angle is a vector, returns a 3x3xN rotation matrix.
    """
    if not _axis_already_normalized:
        axis = axis / np.linalg.norm(axis)

    sintheta = np.sin(angle)
    costheta = np.cos(angle)
    cpm = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])  # The cross product matrix of the rotation axis vector
    outer_axis = axis @ axis.T

    rot_matrix = costheta * np.eye(3) + sintheta * cpm + (1 - costheta) * outer_axis
    return rot_matrix


def linspace_3D(start, stop, n_points):  # TODO make n-dimensional
    """
    Given two points (a start and an end), returns an interpolated array of points on the line between the two.
    :param start: 3D coordinates expressed as a 1D numpy array, shape==(3).
    :param stop: 3D coordinates expressed as a 1D numpy array, shape==(3).
    :param n_points: Number of points to be interpolated (including endpoints), a scalar.
    :return: Array of 3D coordinates expressed as a 2D numpy array, shape==(N, 3)
    """
    x = cas.linspace(start[0], stop[0], n_points)
    y = cas.linspace(start[1], stop[1], n_points)
    z = cas.linspace(start[2], stop[2], n_points)

    points = cas.horzcat(x, y, z)
    return points
