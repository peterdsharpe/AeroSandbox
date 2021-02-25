import casadi as cas


def is_casadi_type(array_like, recursive=True) -> bool:
    """
    Returns a boolean of whether an object is a CasADi data type or not. If the recursive flag is True, iterates recursively.

    Args:

        object: The object to evaluate.

        recursive: If the object is a list or tuple, recursively iterate through every subelement. If any of the
        subelements are a CasADi type, return True. Otherwise, return False

    Returns: A boolean if the object is a CasADi data type.

    """
    if recursive and (isinstance(array_like, list) or isinstance(array_like, tuple)):
        for element in array_like:
            if is_casadi_type(element, recursive=True):
                return True

    return (
            isinstance(array_like, cas.MX) or
            isinstance(array_like, cas.DM) or
            isinstance(array_like, cas.SX)
    )
