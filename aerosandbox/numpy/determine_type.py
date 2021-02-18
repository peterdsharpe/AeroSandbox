import casadi as cas


def is_casadi_type(object) -> bool:
    """
    Returns a boolean of whether an object is a CasADi data type or not.
    Args:
        object: The object to evaluate.

    Returns: A boolean if the object is a CasADi data type.

    """
    return (
            isinstance(object, cas.MX) or
            isinstance(object, cas.DM) or
            isinstance(object, cas.SX)
    )
