# A mix of numpy interpolation routines and scipy.interpolate.

import numpy as _onp
import casadi as _cas
from aerosandbox.numpy.determine_type import is_casadi_type
from aerosandbox.numpy.array import array
from aerosandbox.numpy.conditionals import where
from aerosandbox.numpy.logicals import all, any, logical_or


def interp(x, xp, fp, left=None, right=None, period=None):
    """
    One-dimensional linear interpolation, analogous to numpy.interp().

    Returns the one-dimensional piecewise linear interpolant to a function with given discrete data points (xp, fp),
    evaluated at x.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.interp.html

    Specific notes: xp is assumed to be sorted.
    """
    if not is_casadi_type([x, xp, fp], recursive=True):
        return _onp.interp(
            x=x,
            xp=xp,
            fp=fp,
            left=left,
            right=right,
            period=period
        )

    else:
        ### If xp or x are CasADi types, this is unsupported :(
        if is_casadi_type([x, xp], recursive=True):
            raise NotImplementedError(
                "Unfortunately, CasADi doesn't yet support a dispatch for x or xp as CasADi types."
            )

        ### Handle period argument
        if period is not None:
            if any(
                    logical_or(
                        xp < 0,
                        xp > period
                    )
            ):
                raise NotImplementedError(
                    "Haven't yet implemented handling for if xp is outside the period.")  # Not easy to implement because casadi doesn't have a sort feature.

            x = _cas.mod(x, period)

        ### Make sure x isn't an int
        if isinstance(x, int):
            x = float(x)

        ### Make sure that x is an iterable
        try:
            x[0]
        except TypeError:
            x = array([x], dtype=float)

        ### Make sure xp is an iterable
        xp = array(xp, dtype=float)

        ### Do the interpolation
        f = _cas.interp1d(
            xp,
            fp,
            x
        )

        ### Handle left/right
        if left is not None:
            f = where(
                x < xp[0],
                left,
                f
            )
        if right is not None:
            f = where(
                x > xp[-1],
                right,
                f
            )

        ### Return
        return f
