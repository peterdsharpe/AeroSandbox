import casadi as cas
from typing import Tuple, Union


def sind(x):
    return cas.sin(x * cas.pi / 180)


def cosd(x):
    return cas.cos(x * cas.pi / 180)


def tand(x):
    return cas.tan(x * cas.pi / 180)


def atan2d(y, x):
    return cas.atan2(y, x) * 180 / np.pi


def clip(x, min, max):  # Clip a value to a range [min, max].
    return cas.fmin(cas.fmax(min, x), max)


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
            *
            *
            *
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
        s = cas.tanh(x)
    elif sigmoid_type == "arctan":
        s = 2 / cas.pi * cas.arctan(cas.pi / 2 * x)
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
        value_switch_low,
        value_switch_high,
        switch: float
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
        value_switch_low: Value to be returned when switch is low. Can be a float or an array.
        value_switch_high: Value to be returned when switch is high. Can be a float or an array.
        switch: A value that acts as a "switch" between the two values [float].
            If switch is -Inf, value_switch_low is returned.
            If switch is Inf, value_switch_high is returned.
            If switch is 0, the mean of value_switch_low and value_switch_high is returned.
            If switch is 1, the return value is roughly (0.88 * value_switch_high + 0.12 * value_switch_low).
            If switch is -1, the return value is roughly (0.88 * value_switch_low + 0.12 * value_switch_high).

    Returns: A value that is a blend between value_switch_low and value_switch_high, with the weighting dependent
        on the value of the 'switch' parameter.

    """
    blend_function = lambda x: sigmoid(
        x,
        sigmoid_type="tanh",
        normalization_range=(0, 1)
    )
    weight_to_value_switch_high = blend_function(switch)

    blend_value = (
            value_switch_high * weight_to_value_switch_high +
            value_switch_low * (1 - weight_to_value_switch_high)
    )

    return blend_value


def smoothmax(value1, value2, hardness):
    """
    A smooth maximum between two functions.
    Useful because it's differentiable and convex!
    Great writeup by John D Cook here:
        https://www.johndcook.com/soft_maximum.pdf
    :param value1: Value of function 1.
    :param value2: Value of function 2.
    :param hardness: Hardness parameter. Higher values make this closer to max(x1, x2).
    :return: Soft maximum of the two supplied values.
    """
    value1 = value1 * hardness
    value2 = value2 * hardness
    max = cas.fmax(value1, value2)
    min = cas.fmin(value1, value2)
    out = max + cas.log(1 + cas.exp(min - max))
    out /= hardness
    return out


if __name__ == '__main__':
    import numpy as np

    # Test smoothmax
    import matplotlib.pyplot as plt
    from matplotlib import style
    import seaborn as sns

    sns.set(font_scale=1)
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
    x = np.linspace(-10, 10, 100)
    y1 = x
    y2 = -2 * x - 3
    hardness = 0.5
    plt.plot(x, y1, label="y1")
    plt.plot(x, y2, label="y2")
    plt.plot(x, smoothmax(y1, y2, hardness), label="smoothmax")
    plt.xlabel(r"x")
    plt.ylabel(r"y")
    plt.title(r"Smoothmax")
    plt.tight_layout()
    plt.legend()
    # plt.savefig("C:/Users/User/Downloads/temp.svg")
    plt.show()
