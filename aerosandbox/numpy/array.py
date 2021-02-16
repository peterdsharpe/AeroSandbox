import numpy as onp
import casadi as cas
from aerosandbox.numpy.determine_type import is_casadi_type


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


def roll(a, shift, axis=0):
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
    if not is_casadi_type(a):
        return onp.roll(a, shift, axis=axis)
    else:  #TODO add some checking to make sure shift < len(a)
        # assert shift < a.shape[axis]
        if 1 in a.shape and axis==0:
            return cas.vertcat(a[-shift, :], a[:-shift, :])
        elif axis == 1:
            return cas.horzcat(a[:, -shift], a[:, :-shift])
        elif axis == 0:
            return cas.vertcat(a.T[:, -shift], a.T[:, :-shift]).T
        else:
            raise Exception("Wrong axis")
        
        
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
    
def reshape(a, newshape):
    """Gives a new shape to an array without changing its data."""
    
    if not is_casadi_type(a):
        return onp.reshape(a, newshape)
    else:
        return cas.reshape(a, newshape)