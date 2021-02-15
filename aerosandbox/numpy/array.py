import numpy as onp
import casadi as cas


def array(object, dtype=None):
    a = onp.array(object, dtype=dtype)

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
    except TypeError:  # array has no function len() -> either float, int, or CasADi type
        try:
            if len(array.shape) >= 1:
                return array.shape[0]
            else:
                raise AttributeError
        except AttributeError:  # array has no attribute shape -> either float or int
            return 1


def roll(a, shift):
    """
    Roll array elements along a given axis.

    Elements that roll beyond the last position are re-introduced at the first.

    Parameters
    ----------
    a : array_like
        Input array.
    shift : int
        The number of places by which elements are shifted.

    Returns
    -------
    res : ndarray
        Output array, with the same shape as a.

    """
    try:
        return onp.roll()
    except Exception:
        if len(a.shape) == 1:
            return cas.vertcat(a[-shift], a[:-shift])
        else:
            return cas.vertcat(a[-shift, :], a[:-shift, :])
        
        
def max(a):
    """
    Returns the maximum value of an array
    """
            
    try: 
        return onp.max(a)
    except TypeError:
        return cas.mmax(a)


def min(a):
    """
    Returns the minimum value of an array
    """
            
    try:
        return onp.min(a)
    except TypeError:
        return cas.mmin(a)