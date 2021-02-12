import numpy as onp
import casadi as cas


def array(object, dtype=None):
    try:
        a = onp.array(object, dtype=dtype)
        if a.dtype == "O":
            raise Exception
        return a
    except (AttributeError, Exception): # If this occurs, it needs to be a CasADi type.
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
    Returns the length of an 1D-array-like object.
    Args:
        array:

    Returns:

    """
    try:
        return len(array)
    except TypeError:  # array has no function len() -> either float, int, or CasADi type
        try:
            if len(array.shape) >= 1:
                return array.shape[0]
            else:
                raise AttributeError
        except AttributeError:  # array has no attribute shape -> either float or int
            return 1
