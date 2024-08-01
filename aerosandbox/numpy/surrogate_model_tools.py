import aerosandbox.numpy as _np
import casadi as _cas
from typing import Tuple, Union


def softmax(
    *args: Union[float, _np.ndarray],
    softness: float = None,
    hardness: float = None,
) -> Union[float, _np.ndarray]:
    """
    An element-wise softmax between two or more arrays. Also referred to as the logsumexp() function.

    Useful for optimization because it's differentiable and preserves convexity!

    Great writeup by John D Cook here:
        https://www.johndcook.com/soft_maximum.pdf

    Notes: Can provide either `hardness` or `softness`, not both. These are the inverse of each other. If neither is
    provided, `hardness` is set to 1.

    Args:

        *args: Provide any number of arguments as values to take the softmax of.

        hardness: Hardness parameter. Higher values make this closer to max(x1, x2).

        softness: Softness parameter. (Inverse of hardness.) Lower values make this closer to max(x1, x2).

            - Setting `softness` is particularly useful, because it has the same units as each of the function's
            inputs. For example, if you're taking the softmax of two values that are lengths in units of meters,
            then `softness` is also in units of meters. In this case, `softness` has the rough meaning of "an amount
            of discrepancy between the input values that would be considered physically significant".

    Returns:
        Soft maximum of the supplied values.
    """
    ### Set defaults for hardness/softness
    n_specified_arguments = (hardness is not None) + (softness is not None)
    if n_specified_arguments == 0:
        softness = 1
    elif n_specified_arguments == 2:
        raise ValueError("You must provide exactly one of `hardness` or `softness`.")

    if hardness is not None:
        softness = 1 / hardness

    if _np.any(softness <= 0):
        if softness is not None:
            raise ValueError("The value of `softness` must be positive.")
        else:
            raise ValueError("The value of `hardness` must be positive.")

    if len(args) <= 1:
        raise ValueError(
            "You must call softmax with the value of two or more arrays that you'd like to take the "
            "element-wise softmax of."
        )

    ### Scale the args by softness
    args = [arg / softness for arg in args]

    ### Find the element-wise max and min of the arrays:
    min = args[0]
    max = args[0]
    for arg in args[1:]:
        min = _np.fmin(min, arg)
        max = _np.fmax(max, arg)

    out = max + _np.log(
        sum([_np.exp(_np.maximum(array - max, -500)) for array in args])
    )
    out = out * softness
    return out


def softmin(
    *args: Union[float, _np.ndarray],
    softness: float = None,
    hardness: float = None,
) -> Union[float, _np.ndarray]:
    """
    An element-wise softmin between two or more arrays. Related to the logsumexp() function.

    Useful for optimization because it's differentiable and preserves convexity!

    Great writeup by John D Cook here:
        https://www.johndcook.com/soft_maximum.pdf

    Notes: Can provide either `hardness` or `softness`, not both. These are the inverse of each other. If neither is
    provided, `hardness` is set to 1.

    Args:

        *args: Provide any number of arguments as values to take the softmin of.

        hardness: Hardness parameter. Higher values make this closer to min(x1, x2).

        softness: Softness parameter. (Inverse of hardness.) Lower values make this closer to min(x1, x2).

            - Setting `softness` is particularly useful, because it has the same units as each of the function's
            inputs. For example, if you're taking the softmin of two values that are lengths in units of meters,
            then `softness` is also in units of meters. In this case, `softness` has the rough meaning of "an amount
            of discrepancy between the input values that would be considered physically significant".

    Returns:
        Soft minimum of the supplied values.
    """
    return -softmax(
        *[-arg for arg in args],
        softness=softness,
        hardness=hardness,
    )


def softmax_scalefree(
    *args: Union[float, _np.ndarray],
    relative_softness: float = None,
    relative_hardness: float = None,
) -> Union[float, _np.ndarray]:
    n_specified_arguments = (relative_hardness is not None) + (
        relative_softness is not None
    )
    if n_specified_arguments == 0:
        relative_softness = 0.01
    elif n_specified_arguments == 2:
        raise ValueError(
            "You must provide exactly one of `relative_softness` or `relative_hardness."
        )

    if relative_hardness is not None:
        relative_softness = 1 / relative_hardness

    return softmax(*args, softness=relative_softness * _np.linalg.norm(_np.array(args)))


def softmin_scalefree(
    *args: Union[float, _np.ndarray],
    relative_softness: float = None,
    relative_hardness: float = None,
) -> Union[float, _np.ndarray]:
    return -softmax_scalefree(
        *[-arg for arg in args],
        relative_softness=relative_softness,
        relative_hardness=relative_hardness,
    )


def softplus(
    x: Union[float, _np.ndarray],
    beta=1,
    threshold=40,
):
    """
    A smooth approximation of the ReLU function, applied elementwise to an array `x`.

    Softplus(x) = 1/beta * log(1 + exp(beta * x))

    Often used as an activation function in neural networks.

    Args:
        x: The input
        beta: A parameter that controls the "softness" of the function. Higher values of beta make the function
            approach ReLU.
        threshold: Values above this threshold are approximated as linear.

    Returns: The value of the softplus function.
    """
    if _np.is_casadi_type(x, recursive=False):
        return _np.where(
            beta * x > threshold, x, 1 / beta * _cas.log1p(_cas.exp(beta * x))
        )
    else:
        return 1 / beta * _np.logaddexp(0, beta * x)


def sigmoid(
    x,
    sigmoid_type: str = "tanh",
    normalization_range: Tuple[Union[float, int], Union[float, int]] = (0, 1),
):
    """
    A sigmoid function. From Wikipedia (https://en.wikipedia.org/wiki/Sigmoid_function):
        A sigmoid function is a mathematical function having a characteristic "S"-shaped curve
        or sigmoid curve.

    Args:
        x: The input
        sigmoid_type: Type of sigmoid function to use [str]. Can be one of:
            * "tanh" or "logistic" (same thing)
            * "arctan"
            * "polynomial"
        normalization_type: Range in which to normalize the sigmoid, shorthanded here in the
            documentation as "N". This parameter is given as a two-element tuple (min, max).

            After normalization:
                >>> sigmoid(-Inf) == normalization_range[0]
                >>> sigmoid(Inf) == normalization_range[1]

            * In the special case of N = (0, 1):
                >>> sigmoid(-Inf) == 0
                >>> sigmoid(Inf) == 1
                >>> sigmoid(0) == 0.5
                >>> d(sigmoid)/dx at x=0 == 0.5
            * In the special case of N = (-1, 1):
                >>> sigmoid(-Inf) == -1
                >>> sigmoid(Inf) == 1
                >>> sigmoid(0) == 0
                >>> d(sigmoid)/dx at x=0 == 1

    Returns: The value of the sigmoid.
    """
    ### Sigmoid equations given here under the (-1, 1) normalization:
    if sigmoid_type == ("tanh" or "logistic"):
        # Note: tanh(x) is simply a scaled and shifted version of a logistic curve.
        s = _np.tanh(x)
    elif sigmoid_type == "arctan":
        s = 2 / _np.pi * _np.arctan(_np.pi / 2 * x)
    elif sigmoid_type == "polynomial":
        s = x / (1 + x**2) ** 0.5
    else:
        raise ValueError("Bad value of parameter 'type'!")

    ### Normalize
    min = normalization_range[0]
    max = normalization_range[1]
    s_normalized = s * (max - min) / 2 + (max + min) / 2

    return s_normalized


def swish(
    x: Union[float, _np.ndarray],
    beta: float = 1.0,
):
    """
    A smooth approximation of the ReLU function, applied elementwise to an array `x`.

    Swish(x) = x / (1 + exp(-beta * x)) = x * logistic(x) = x * (0.5 + 0.5 * tanh(x/2))

    Often used as an activation function in neural networks.

    Args:
        x: The input
        beta: A parameter that controls the "softness" of the function. Higher values of beta make the function
            approach ReLU.

    Returns: The value of the swish function.
    """
    return x / (1 + _np.exp(-beta * x))


def blend(
    switch: float,
    value_switch_high,
    value_switch_low,
):
    """
    Smoothly blends between two values on the basis of some switch function.

    This function is similar in usage to numpy.where (documented here:
    https://numpy.org/doc/stable/reference/generated/numpy.where.html) , except that
    instead of using a boolean as to switch between the two values, a float is used to
    smoothly transition between the two in a differentiable manner.

    Before using this function, be sure to understand the difference between this and
    smoothmax(), and choose the correct one.

    Args:
        switch: A value that acts as a "switch" between the two values [float].
            If switch is -Inf, value_switch_low is returned.
            If switch is Inf, value_switch_high is returned.
            If switch is 0, the mean of value_switch_low and value_switch_high is returned.
            If switch is 1, the return value is roughly (0.88 * value_switch_high + 0.12 * value_switch_low).
            If switch is -1, the return value is roughly (0.88 * value_switch_low + 0.12 * value_switch_high).
        value_switch_high: Value to be returned when switch is high. Can be a float or an array.
        value_switch_low: Value to be returned when switch is low. Can be a float or an array.

    Returns: A value that is a blend between value_switch_low and value_switch_high, with the weighting dependent
        on the value of the 'switch' parameter.

    """
    def blend_function(x):
        return sigmoid(x, normalization_range=(0, 1))
    weight_to_value_switch_high = blend_function(switch)

    blend_value = value_switch_high * weight_to_value_switch_high + value_switch_low * (
        1 - weight_to_value_switch_high
    )

    return blend_value
