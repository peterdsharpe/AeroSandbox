import copy
from aerosandbox.tools.string_formatting import eng_string
from aerosandbox.tools.casadi_functions import *
import numpy as np

try:
    from xfoil import XFoil
    from xfoil import model as xfoil_model
except ModuleNotFoundError:
    pass



class AeroSandboxObject:
    def substitute_solution(self, sol):
        """
        Substitutes a solution from CasADi's solver.
        :param sol:
        :return:
        """
        for attrib_name in dir(self):
            attrib_orig = getattr(self, attrib_name)
            if isinstance(attrib_orig, bool) or isinstance(attrib_orig, int):
                continue
            try:
                setattr(self, attrib_name, sol.value(attrib_orig))
            except NotImplementedError:
                pass
            if isinstance(attrib_orig, list):
                try:
                    new_attrib_orig = []
                    for item in attrib_orig:
                        new_attrib_orig.append(item.substitute_solution(sol))
                    setattr(self, attrib_name, new_attrib_orig)
                except:
                    pass
            try:
                setattr(self, attrib_name, attrib_orig.substitute_solution(sol))
            except:
                pass
        return self


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


def cosspace(min=0, max=1, n_points=50):
    """
    Returns cosine-spaced points using CasADi. Syntax analogous to np.linspace().
    :param min: Minimum value
    :param max: Maximum value
    :param n_points: Number of points
    :return: CasADi array
    """
    mean = (max + min) / 2
    amp = (max - min) / 2
    return mean + amp * cas.cos(cas.linspace(cas.pi, 0, n_points))


def np_cosspace(min=0, max=1, n_points=50):
    """
    Returns cosine-spaced points using NumPy. Syntax analogous to np.linspace().
    :param min: Minimum value
    :param max: Maximum value
    :param n_points: Number of points
    :return: 1D NumPy array
    """
    mean = (max + min) / 2
    amp = (max - min) / 2
    return mean + amp * np.cos(np.linspace(np.pi, 0, n_points))


def angle_axis_rotation_matrix(angle, axis, axis_already_normalized=False):
    """
    Gives the rotation matrix from an angle and an axis.
    An implmentation of https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    :param angle: can be one angle or a vector (1d ndarray) of angles. Given in radians.
    :param axis: a 1d numpy array of length 3 (p,y,z). Represents the angle.
    :param axis_already_normalized: boolean, skips normalization for speed if you flag this true.
    :return:
        * If angle is a scalar, returns a 3x3 rotation matrix.
        * If angle is a vector, returns a 3x3xN rotation matrix.
    """
    if not axis_already_normalized:
        axis = axis / cas.norm_2(axis)

    sintheta = cas.sin(angle)
    costheta = cas.cos(angle)
    cpm = cas.vertcat(
        cas.horzcat(0, -axis[2], axis[1]),
        cas.horzcat(axis[2], 0, -axis[0]),
        cas.horzcat(-axis[1], axis[0], 0),
    )  # The cross product matrix of the rotation axis vector
    outer_axis = axis @ cas.transpose(axis)

    rot_matrix = costheta * cas.DM.eye(3) + sintheta * cpm + (1 - costheta) * outer_axis
    return rot_matrix


def linspace_3D(start, stop, n_points):
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