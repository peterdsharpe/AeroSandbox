import aerosandbox.numpy as np
from aerosandbox.numpy.determine_type import is_casadi_type


def reflect_over_XZ_plane(input_vector):
    """
    Takes in a vector or an array and flips the y-coordinates.
    :param input_vector: A vector or list of vectors to flip.
    :return: Vector with flipped sign on y-coordinate.
    """
    if not is_casadi_type(input_vector):
        shape = input_vector.shape
        if len(shape) == 1:
            return input_vector * np.array([1, -1, 1])
        elif len(shape) == 2:
            if not shape[1] == 3:
                raise ValueError("The function expected either a 3-element vector or a Nx3 array!")
            return input_vector * np.array([1, -1, 1])
        else:
            raise ValueError("The function expected either a 3-element vector or a Nx3 array!")

    else:
        if input_vector.shape[1] == 1:
            return input_vector * np.array([1, -1, 1])
        elif input_vector.shape[1] == 3:
            return np.stack((
                input_vector[:, 0],
                -1 * input_vector[:, 1],
                input_vector[:, 2],
            ), axis=1)
        else:
            raise ValueError("This function expected either a 3-element vector or an Nx3 array!")
