import numpy as _onp
import casadi as _cas
from aerosandbox.numpy.determine_type import is_casadi_type


def where(
        condition,
        value_if_true,
        value_if_false,
):
    if not is_casadi_type([condition, value_if_true, value_if_false], recursive=True):
        return _onp.where(
            condition,
            value_if_true,
            value_if_false
        )
    else:
        return _cas.if_else(
            condition,
            value_if_true,
            value_if_false
        )
