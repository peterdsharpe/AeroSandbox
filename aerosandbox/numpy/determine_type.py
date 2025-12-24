"""Type detection utilities for NumPy/CasADi dual-backend support.

This module provides functions to determine whether objects are CasADi symbolic
types, enabling runtime dispatch between NumPy and CasADi backends.
"""
import numpy as _onp
import casadi as _cas
from typing import Any


def is_casadi_type(object: Any, recursive: bool = True) -> bool:
    """Check whether an object is or contains a CasADi data type.

    Parameters
    ----------
    object : Any
        The object to evaluate.
    recursive : bool, optional
        If True and the object is a list, tuple, set, or dict, recursively
        iterate through every subelement. Return True if any subelement (at
        any depth) is a CasADi type. Default is True.

    Returns
    -------
    bool
        True if the object is (or contains, if ``recursive=True``) a CasADi
        data type (MX, DM, or SX). False otherwise.

    Examples
    --------
    >>> import casadi as cas
    >>> is_casadi_type(cas.MX.sym("x"))
    True
    >>> is_casadi_type([1, 2, cas.MX.sym("x")], recursive=True)
    True
    >>> is_casadi_type([1, 2, 3], recursive=True)
    False
    """
    t = type(object)

    # NumPy arrays cannot be or contain CasADi types, unless they are object arrays
    if t == _onp.ndarray and object.dtype != "O":
        return False

    # Skip certain Python types known not to be or contain CasADi types.
    for type_to_skip in (
        float,
        int,
        complex,
        bool,
        str,
        range,
        type(None),
        bytes,
        bytearray,
        memoryview,
    ):
        if t == type_to_skip:
            return False

    # If it's directly a CasADi type, we're done.
    if t == _cas.MX or t == _cas.DM or t == _cas.SX:
        return True

    # At this point, we know it's not a CasADi type, but we don't know if it *contains* a CasADi type (relevant if recursing)
    if recursive:
        if (
            issubclass(t, list)
            or issubclass(t, tuple)
            or issubclass(t, set)
            or (t == _onp.ndarray and object.dtype == "O")
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


def is_iterable(x: Any) -> bool:
    """Check whether an object is iterable.

    Parameters
    ----------
    x : Any
        The object to evaluate.

    Returns
    -------
    bool
        True if the object is iterable (has an ``__iter__`` method),
        False otherwise.
    """
    try:
        iter(x)
        return True
    except TypeError:
        return False
