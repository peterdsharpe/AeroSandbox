import numpy as np
import casadi as cas


def where(
        condition,
        value_if_true,
        value_if_false,
):
    try:
        return np.where(
            condition,
            value_if_true,
            value_if_false
        )
    except Exception:
        return cas.if_else(
            condition,
            value_if_true,
            value_if_false
        )
