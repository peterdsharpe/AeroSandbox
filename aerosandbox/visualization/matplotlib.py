from typing import Callable, Tuple, Union
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
plt.ion()
sns.set(font_scale=1)

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