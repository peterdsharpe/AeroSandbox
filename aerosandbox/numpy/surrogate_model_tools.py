import aerosandbox.numpy as _np
from typing import Tuple, Union


def softmax(*args, hardness=1):
    """
    An element-wise softmax between two or more arrays. Also referred to as the logsumexp() function.

    Useful for optimization because it's differentiable and preserves convexity!

    Great writeup by John D Cook here:
        https://www.johndcook.com/soft_maximum.pdf

    Args:
        Provide any number of arguments as values to take the softmax of.

        hardness: Hardness parameter. Higher values make this closer to max(x1, x2).

    Returns:
        Soft maximum of the supplied values.
    """
    if hardness <= 0:
        raise ValueError("The value of `hardness` must be positive.")

    if len(args) <= 1:
        raise ValueError("You must call softmax with the value of two or more arrays that you'd like to take the "
                         "element-wise softmax of.")

    ### Scale the args by hardness
    args = [arg * hardness for arg in args]

    ### Find the element-wise max and min of the arrays:
    min = args[0]
    max = args[0]
    for arg in args[1:]:
        min = _np.fmin(min, arg)
        max = _np.fmax(max, arg)

    out = max + _np.log(sum(
            [_np.exp(array - max) for array in args]
        )
    )
    out = out / hardness
    return out


def sigmoid(
        x,
        sigmoid_type: str = "tanh",
        normalization_range: Tuple[Union[float, int], Union[float, int]] = (0, 1)
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
        # Note: tanh(x) is simply a scaled and shifted version of a logistic curve; after
        #   normalization these functions are identical.
        s = _np.tanh(x)
    elif sigmoid_type == "arctan":
        s = 2 / _np.pi * _np.arctan(_np.pi / 2 * x)
    elif sigmoid_type == "polynomial":
        s = x / (1 + x ** 2) ** 0.5
    else:
        raise ValueError("Bad value of parameter 'type'!")

    ### Normalize
    min = normalization_range[0]
    max = normalization_range[1]
    s_normalized = s * (max - min) / 2 + (max + min) / 2

    return s_normalized


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
    blend_function = lambda x: sigmoid(
        x,
        normalization_range=(0, 1)
    )
    weight_to_value_switch_high = blend_function(switch)

    blend_value = (
            value_switch_high * weight_to_value_switch_high +
            value_switch_low * (1 - weight_to_value_switch_high)
    )

    return blend_value
