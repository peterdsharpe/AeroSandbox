"""Top-level linear algebra functions for the AeroSandbox NumPy-like interface.

This module provides linear algebra functions that live in the top-level
namespace (rather than under ``linalg``), working with both NumPy arrays and
CasADi symbolic arrays.
"""

import numpy as _onp
import casadi as _cas
from aerosandbox.numpy.determine_type import is_casadi_type


def dot(a, b, manual=False):
    """Compute the dot product of two arrays.

    Parameters
    ----------
    a
        First input array.
    b
        Second input array.
    manual : bool, optional
        If True, use a manual loop implementation. Default is False.

    Returns
    -------
    Scalar | Array
        The dot product of ``a`` and ``b``.

    See Also
    --------
    numpy.dot : https://numpy.org/doc/stable/reference/generated/numpy.dot.html
    """
    if manual:
        return sum([ai * bi for ai, bi in zip(a, b)])

    if not is_casadi_type([a, b], recursive=True):
        return _onp.dot(a, b)

    else:
        return _cas.dot(a, b)


def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None, manual=False):
    """Return the cross product of two (arrays of) vectors.

    Parameters
    ----------
    a
        Components of the first vector(s).
    b
        Components of the second vector(s).
    axisa : int, optional
        Axis of ``a`` that defines the vector(s). Default is -1 (the last axis).
        For CasADi arrays, must be -1, 0, or 1.
    axisb : int, optional
        Axis of ``b`` that defines the vector(s). Default is -1 (the last axis).
        For CasADi arrays, must be -1, 0, or 1.
    axisc : int, optional
        Axis of the result containing the cross product vector(s). Default is -1
        (the last axis). For CasADi arrays, must be -1, 0, or 1.
    axis : int, optional
        If given, the axis of ``a``, ``b``, and the result that defines the
        vector(s) and cross product(s). Overrides ``axisa``, ``axisb``, and
        ``axisc``.
    manual : bool, optional
        If True, use a manual implementation that treats ``a`` and ``b`` as
        3-element sequences and returns a tuple of the three cross-product
        components. Default is False.

    Returns
    -------
    Array
        Vector cross product(s). If ``manual`` is True, a tuple of the three
        components is returned instead.

    See Also
    --------
    numpy.cross : https://numpy.org/doc/stable/reference/generated/numpy.cross.html
    """
    if manual:
        cross_x = a[1] * b[2] - a[2] * b[1]
        cross_y = a[2] * b[0] - a[0] * b[2]
        cross_z = a[0] * b[1] - a[1] * b[0]
        return cross_x, cross_y, cross_z

    if not is_casadi_type([a, b], recursive=True):
        return _onp.cross(a, b, axisa=axisa, axisb=axisb, axisc=axisc, axis=axis)

    else:
        if axis is not None:
            if not (axis == -1 or axis == 0 or axis == 1):
                raise ValueError("`axis` must be -1, 0, or 1.")
            axisa = axis
            axisb = axis
            axisc = axis

        if axisa == -1 or axisa == 1:
            if not is_casadi_type(a):
                a = _cas.DM(a)
            a = a.T
        elif axisa == 0:
            pass
        else:
            raise ValueError("`axisa` must be -1, 0, or 1.")

        if axisb == -1 or axisb == 1:
            if not is_casadi_type(b):
                b = _cas.DM(b)
            b = b.T
        elif axisb == 0:
            pass
        else:
            raise ValueError("`axisb` must be -1, 0, or 1.")

        # Compute the cross product, horizontally (along axis 1 of a 2D array)
        c = _cas.cross(a, b)

        if axisc == -1 or axisc == 1:
            c = c.T
        elif axisc == 0:
            pass
        else:
            raise ValueError("`axisc` must be -1, 0, or 1.")

        return c


def transpose(a, axes=None):
    """Reverse or permute the axes of an array; return the modified array.

    For an array ``a`` with two axes, ``transpose(a)`` gives the matrix transpose.

    Parameters
    ----------
    a
        Input array.
    axes : tuple or list of ints, optional
        Permutation of the axes (NumPy backend only). If None (default), the
        order of the axes is reversed.

    Returns
    -------
    Array
        The input array with its axes permuted.

    See Also
    --------
    numpy.transpose : https://numpy.org/doc/stable/reference/generated/numpy.transpose.html
    """
    if not is_casadi_type(a, recursive=False):
        return _onp.transpose(a, axes=axes)
    else:
        return _cas.transpose(a)
