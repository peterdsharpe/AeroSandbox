from typing import Callable, Tuple, Union
import numpy as np
import casadi as cas
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
plt.ion()
sns.set(font_scale=1)

#
# def reflect_over_XZ_plane(input_vector):
#     # Takes in a vector or an array and flips the y-coordinates.
#     output_vector = input_vector
#     shape = output_vector.shape
#     if len(shape) == 1 and shape[0] == 3:
#         output_vector = output_vector * cas.vertcat(1, -1, 1)
#     elif len(shape) == 2 and shape[1] == 1 and shape[0] == 3:  # Vector of 3 items
#         output_vector = output_vector * cas.vertcat(1, -1, 1)
#     elif len(shape) == 2 and shape[1] == 3:  # 2D Nx3 vector
#         output_vector = cas.horzcat(output_vector[:, 0], -1 * output_vector[:, 1], output_vector[:, 2])
#     # elif len(shape) == 3 and shape[2] == 3:  # 3D MxNx3 vector
#     #     output_vector = output_vector * cas.array([1, -1, 1])
#     else:
#         raise Exception("Invalid input for reflect_over_XZ_plane!")
#
#     return output_vector
#



def contour(
        func: Callable,
        x_range: Tuple[Union[float,int],Union[float,int]],
        y_range: Tuple[Union[float,int],Union[float,int]],
        resolution:int =50,
        show:bool=True,  # type: bool
):
    """
    Makes a contour plot of a function of 2 variables. Can also plot a list of functions.
    :param func: function of form f(x,y) to plot.
    :param x_range: Range of x values to plot, expressed as a tuple (x_min, x_max)
    :param y_range: Range of y values to plot, expressed as a tuple (y_min, y_max)
    :param resolution: Resolution in x and y to plot. [int]
    :param show: Should we show the plot?
    :return:
    """
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    # TODO finish function