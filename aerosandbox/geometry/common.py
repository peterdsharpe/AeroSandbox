
import copy
from aerosandbox.visualization import *
from aerosandbox.tools.string_formatting import *
from aerosandbox.tools.casadi_tools import *

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


def plot_point_cloud(
        p  # type: np.ndarray
):
    """
    Plots an Nx3 point cloud with Plotly
    :param p: An Nx3 array of points to be plotted.
    :return: None
    """
    p = np.array(p)
    px.scatter_3d(x=p[:, 0], y=p[:, 1], z=p[:, 2]).show()


def kulfan_coordinates(
        lower_weights=-0.2 * np.ones(5),  # type: np.ndarray
        upper_weights=0.2 * np.ones(5),  # type: np.ndarray
        enforce_continuous_LE_radius=True,
        TE_thickness=0.005,  # type: float
        n_points_per_side=100,  # type: int
        N1=0.5,  # type: float
        N2=1.0,  # type: float
):
    """
    Calculates the coordinates of a Kulfan (CST) airfoil.
    To make a Kulfan (CST) airfoil, use the following syntax:

    asb.Airfoil("My Airfoil Name", coordinates = asb.kulfan_coordinates(*args))

    More on Kulfan (CST) airfoils: http://brendakulfan.com/docs/CST2.pdf
    Notes on N1, N2 (shape factor) combinations:
        * 0.5, 1: Conventional airfoil
        * 0.5, 0.5: Elliptic airfoil
        * 1, 1: Biconvex airfoil
        * 0.75, 0.75: Sears-Haack body (radius distribution)
        * 0.75, 0.25: Low-drag projectile
        * 1, 0.001: Cone or wedge airfoil
        * 0.001, 0.001: Rectangle, circular duct, or circular rod.
    :param lower_weights:
    :param upper_weights:
    :param enforce_continuous_LE_radius: Enforces a continous leading-edge radius by throwing out the first lower weight.
    :param TE_thickness:
    :param n_points_per_side:
    :param N1: LE shape factor
    :param N2: TE shape factor
    :return:
    """
    from scipy.special import comb

    if enforce_continuous_LE_radius:
        lower_weights[0] = -1 * upper_weights[0]

    x_lower = np_cosspace(0, 1, n_points_per_side)
    x_upper = x_lower[::-1]

    def shape(w, x):
        # Class function
        C = x ** N1 * (1 - x) ** N2

        # Shape function (Bernstein polynomials)
        n = len(w) - 1  # Order of Bernstein polynomials

        K = comb(n, np.arange(n + 1))  # Bernstein polynomial coefficients

        S_matrix = (
                w * K * np.expand_dims(x, 1) ** np.arange(n + 1) *
                np.expand_dims(1 - x, 1) ** (n - np.arange(n + 1))
        )  # Polynomial coefficient * weight matrix
        S = np.sum(S_matrix, axis=1)

        # Calculate y output
        y = C * S
        return y

    y_lower = shape(lower_weights, x_lower)
    y_upper = shape(upper_weights, x_upper)

    # TE thickness
    y_lower -= x_lower * TE_thickness / 2
    y_upper += x_upper * TE_thickness / 2

    x = np.concatenate([x_upper, x_lower])
    y = np.concatenate([y_upper, y_lower])
    coordinates = np.vstack((x, y)).T

    return coordinates
