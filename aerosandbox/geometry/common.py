import aerosandbox.numpy as np
import casadi as cas

def reflect_over_XZ_plane(input_vector):
    """
    Takes in a vector or an array and flips the y-coordinates.
    :param input_vector: A vector or list of vectors to flip.
    :return: Vector with flipped sign on y-coordinate.
    """
    if isinstance(input_vector, np.ndarray):
        shape = input_vector.shape
        if len(shape) == 1:
            return input_vector * np.array([1, -1, 1])
        elif len(shape) == 2:
            if not shape[1] == 3:
                raise ValueError("The function expected either a 3-element vector or a Nx3 array!")
            return input_vector * np.array([1, -1, 1])
        else:
            raise ValueError("The function expected either a 3-element vector or a Nx3 array!")

    if not (
        isinstance(input_vector, cas.MX) or
        isinstance(input_vector, cas.DM) or
        isinstance(input_vector, cas.SX)
    ):
        raise TypeError("Got an unexpected data type for `input_vector`!")

    if input_vector.shape[1] == 1:
        return input_vector * np.array([1, -1, 1])
    elif input_vector.shape[1] == 3:
        return cas.horzcat(
            input_vector[:,0],
            -1 * input_vector[:,1],
            input_vector[:,2],
        )
    else:
        raise ValueError("This function expected either a 3-element vector or an Nx3 array!")