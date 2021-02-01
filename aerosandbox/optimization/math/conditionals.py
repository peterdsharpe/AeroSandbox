import numpy as np
import casadi as cas


def if_else(
        condition,
        value_if_true,
        value_if_false,
):
    try:
        np.where(
            condition,
            value_if_true,
            value_if_false
        )
    except Exception:
        cas.if_else(
            condition,
            value_if_true,
            value_if_false
        )
