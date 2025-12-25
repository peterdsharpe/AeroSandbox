"""Monadic (single-operand) arithmetic functions for AeroSandbox.

This module provides reduction and aggregation functions that work with both
NumPy arrays and CasADi symbolic arrays.
"""
import numpy as _onp
import casadi as _cas
from aerosandbox.numpy.determine_type import is_casadi_type
from aerosandbox.numpy.array import asarray
from aerosandbox.numpy.typing import Array, ArrayLike, Scalar


def sum(x: ArrayLike, axis: int | None = None) -> Scalar | Array:
    """Compute the sum of array elements over a given axis.

    Parameters
    ----------
    x : ArrayLike
        Input array.
    axis : int, optional
        Axis along which to sum. By default (None), sum over all elements.
        For CasADi arrays, only None, 0, or 1 are valid.

    Returns
    -------
    Scalar | Array
        Sum of the elements. If ``axis`` is None, a scalar is returned.

    Raises
    ------
    ValueError
        If CasADi arrays are used with an invalid axis.

    See Also
    --------
    numpy.sum : https://numpy.org/doc/stable/reference/generated/numpy.sum.html
    """
    if not is_casadi_type(x):
        return _onp.sum(x, axis=axis)

    else:
        if axis == 0:
            return _cas.sum1(x).T

        elif axis == 1:
            return _cas.sum2(x)
        elif axis is None:
            return sum(sum(x, axis=0), axis=0)
        else:
            raise ValueError(
                "CasADi types can only be up to 2D, so `axis` must be None, 0, or 1."
            )


def mean(x: ArrayLike, axis: int | None = None) -> Scalar | Array:
    """Compute the arithmetic mean along the specified axis.

    Parameters
    ----------
    x : ArrayLike
        Input array.
    axis : int, optional
        Axis along which to compute the mean. By default (None), compute over
        all elements. For CasADi arrays, only None, 0, or 1 are valid.

    Returns
    -------
    Scalar | Array
        Mean of the elements. If ``axis`` is None, a scalar is returned.

    Raises
    ------
    ValueError
        If CasADi arrays are used with an invalid axis.

    See Also
    --------
    numpy.mean : https://numpy.org/doc/stable/reference/generated/numpy.mean.html
    """
    if not is_casadi_type(x):
        return _onp.mean(x, axis=axis)

    else:
        x = asarray(x)  # Ensure x is Array for .shape access
        if axis == 0:
            return sum(x, axis=0) / x.shape[0]
        elif axis == 1:
            return sum(x, axis=1) / x.shape[1]
        elif axis is None:
            return mean(mean(x, axis=0), axis=1)
        else:
            raise ValueError(
                "CasADi types can only be up to 2D, so `axis` must be None, 0, or 1."
            )


def abs(x: ArrayLike) -> Array:
    """Compute the absolute value element-wise.

    Parameters
    ----------
    x : ArrayLike
        Input array.

    Returns
    -------
    Array
        An array containing the absolute value of each element in ``x``.

    See Also
    --------
    numpy.abs : https://numpy.org/doc/stable/reference/generated/numpy.absolute.html
    """
    if not is_casadi_type(x):
        return _onp.abs(x)

    else:
        return _cas.fabs(x)


# TODO trace()

# def cumsum(x, axis: int = None):
#     """
#     Return the cumulative sum of the elements along a given axis.
#
#     See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html
#     """
#
#     if not is_casadi_type(x):
#         return _onp.cumsum(x, axis=axis)
#
#     else:
#         raise NotImplementedError
#         if axis is None:
#             return _cas.cumsum(_onp.flatten(x))


def prod(x: ArrayLike, axis: int | None = None) -> Scalar | Array:
    """Compute the product of array elements over a given axis.

    Parameters
    ----------
    x : ArrayLike
        Input array.
    axis : int, optional
        Axis along which to compute the product. By default (None), compute
        over all elements. For CasADi arrays, only None, 0, or 1 are valid.

    Returns
    -------
    Scalar | Array
        Product of the elements. If ``axis`` is None, a scalar is returned.

    Notes
    -----
    For CasADi types, this uses a sign-magnitude decomposition to handle
    negative numbers correctly while maintaining O(1) graph complexity::

        prod(x) = exp(sum(log(|x|))) * cos(π * num_negatives)

    where ``num_negatives`` counts elements with negative sign. This correctly
    handles:

    - Positive numbers: standard exp-log identity
    - Negative numbers: sign tracked via cosine of count
    - Zeros: exp(log(0)) = exp(-inf) = 0

    Gradients at x=0 are not well-defined (discontinuous), which is
    mathematically inherent to the product function.

    See Also
    --------
    numpy.prod : https://numpy.org/doc/stable/reference/generated/numpy.prod.html
    """
    if not is_casadi_type(x):
        return _onp.prod(x, axis=axis)

    else:
        ### Compute magnitude: exp(sum(log(|x|)))
        # If any element is 0, log(0) = -inf, sum = -inf, exp(-inf) = 0 (correct)
        abs_x = _cas.fabs(x)
        log_abs_x = _cas.log(abs_x)
        magnitude = _cas.exp(sum(log_abs_x, axis=axis))

        ### Compute sign of product: cos(π * num_negatives)
        # For each element: (1 - sign(x)) / 2 gives 1 if x<0, 0 if x>0, 0.5 if x=0
        # Sum gives count of negatives (plus 0.5 for each zero, but those make magnitude=0)
        # cos(π * n) = 1 if n even, -1 if n odd
        num_negatives = sum((1 - _cas.sign(x)) / 2, axis=axis)
        sign_of_product = _cas.cos(_onp.pi * num_negatives)

        return magnitude * sign_of_product
