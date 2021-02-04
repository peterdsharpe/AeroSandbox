import numpy as np
import casadi as cas


def array(object):
    a = np.array(object)

    if a.dtype != "O":  # If it's not an object array, then you're done here!
        return a

    ### If it is an object array, then we need to make it into a CasADi matrix instead.
    # First, determine the dimension
    def make_row(row):
        try:
            return cas.horzcat(*row)
        except (TypeError, Exception):  # If not iterable or if it's a CasADi MX type
            return row

    return cas.vertcat(
        *[
            make_row(row)
            for row in object
        ]
    )

def length(array) -> int:
    """
    Returns the length of an array-like object.
    Args:
        array:

    Returns:

    """
    try:
        return len(array)
    except TypeError: # array has no function len() -> either float, int, or CasADi type
        try:
            return array.shape[0]
        except AttributeError: # array has no attribute shape -> either float or int
            return 1
