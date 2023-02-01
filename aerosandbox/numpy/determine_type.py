import numpy as _onp
import casadi as _cas
from typing import Any


def is_casadi_type(
        object: Any,
        recursive: bool = True
) -> bool:
    """
    Returns a boolean of whether an object is a CasADi data type or not. If the recursive flag is True,
    iterates recursively, returning True if any subelement (at any depth) is a CasADi type.

    Args:

        object: The object to evaluate.

        recursive: If the object is a list or tuple, recursively iterate through every subelement. If any of the
        subelements (at any depth) are a CasADi type, return True. Otherwise, returns False.

    Returns: A boolean if the object is (or contains, if recursive=True) a CasADi data type.

    """
    t = type(object)

    # NumPy arrays cannot be or contain CasADi types, unless they are object arrays
    if t == _onp.ndarray and object.dtype != 'O':
        return False

    # Skip certain Python types known not to be or contain CasADi types.
    for type_to_skip in (
            float, int, complex,
            bool, str,
            range,
            type(None),
            bytes, bytearray, memoryview
    ):
        if t == type_to_skip:
            return False

    # If it's directly a CasADi type, we're done.
    if (
            t == _cas.MX or
            t == _cas.DM or
            t == _cas.SX
    ):
        return True

    # At this point, we know it's not a CasADi type, but we don't know if it *contains* a CasADi type (relevant if recursing)
    if recursive:
        if (
                issubclass(t, list) or
                issubclass(t, tuple) or
                issubclass(t, set) or
                (
                        t == _onp.ndarray and object.dtype == 'O'
                )
        ):
            for element in object:
                if is_casadi_type(element, recursive=True):
                    return True
            return False

        if issubclass(t, dict):
            for kv in object.items():
                if is_casadi_type(kv, recursive=True):
                    return True
            return False

        return False
    else:
        return False


def is_iterable(x):
    """
    Returns a boolean of whether an object is iterable or not.
    Args:
        x:

    Returns:

    """
    try:
        iter(x)
        return True
    except TypeError:
        return False
