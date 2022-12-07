import matplotlib.pyplot as plt
import matplotlib as mpl
import aerosandbox.numpy as np
from typing import Tuple, Dict, Union, Callable, List
from scipy import interpolate


def plot_smooth(
        *args,
        color=None,
        label=None,
        resample_resolution: int = 500,
        **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Plots a curve that interpolates a 2D dataset. Same as matplotlib.pyplot.plot(), with the following changes:
        * uses B-splines to draw a smooth curve rather than a jagged polyline
        * By default, plots in line format `fmt='.-'` rather than `fmt='-'`.

    Other than that, almost all matplotlib.pyplot.plot() syntax can be used. See syntax here:
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html

    Example usage:
        >>> import aerosandbox.numpy as np
        >>>
        >>> t = np.linspace(0, 1, 12)  # Parametric variable
        >>> x = np.cos(2 * np.pi * t)
        >>> y = np.cos(2 * np.pi * t ** 4) - t
        >>>
        >>> plot_smooth(
        >>>     x, y, 'o--', color='purple'
        >>> )
        >>> plt.show()

    * Note: a true 2D interpolation is performed - it is not assumed y is a function of x, or vice versa. This can,
    in rare cases, cause single-valuedness to not be preserved in cases where it logically should. If this is the
    case, you need to perform the interpolation yourself without `plot_smooth()`.

    Args:

        *args: Same arguments as `matplotlib.pyplot.plot()`.
            Notes on standard plot() syntax:

                Call signatures:
                >>> plot([x], y, [fmt], *, data=None, **kwargs)
                >>> plot([x], y, [fmt], [x2], y2, [fmt2], ..., **kwargs)

                Examples:
                >>> plot(x, y)        # plot x and y using default line style and color
                >>> plot(x, y, 'bo')  # plot x and y using blue circle markers
                >>> plot(y)           # plot y using x as index array 0..N-1
                >>> plot(y, 'r+')     # ditto, but with red plusses

        color: Specifies the color of any line and/or markers that are plotted (as determined by the `fmt`).

        label: Attaches a label to this line. Use `plt.legend()` to display.

        resample_resolution: The number of points to use when resampling the interpolated curve.

        **kwargs: Same keyword arguments as `matplotlib.pyplot.plot()`.

    Returns: A tuple `(x, y)` of the resampled points on the interpolated curve. Both `x` and `y` are 1D ndarrays.

    """
    argslist = list(args)

    if len(args) == 3:
        x = argslist.pop(0)
        y = argslist.pop(0)
        fmt = argslist.pop(0)
    elif len(args) == 2:
        if isinstance(args[1], str):
            x = np.arange(np.length(args[0]))
            y = argslist.pop(0)
            fmt = argslist.pop(0)
        else:
            x = argslist.pop(0)
            y = argslist.pop(0)
            fmt = '.-'
    elif len(args) == 1:
        x = np.arange(np.length(args[0]))
        y = argslist.pop(0)
        fmt = '.-'
    elif len(args) == 0:
        raise ValueError("Missing plot data. Use syntax `plot_smooth(x, y, fmt, *args, **kwargs)'.")

    bspline = interpolate.make_interp_spline(
        x=np.linspace(0, 1, np.length(y)),
        y=np.stack(
            (x, y), axis=1
        )
    )
    result = bspline(np.linspace(0, 1, resample_resolution))
    x_resample = result[:, 0]
    y_resample = result[:, 1]

    scatter_kwargs = {
        **kwargs,
        'linewidth': 0,
    }
    if color is not None:
        scatter_kwargs['color'] = color

    line, = plt.plot(
        x,
        y,
        fmt,
        *argslist,
        **scatter_kwargs
    )

    if color is None:
        color = line.get_color()

    line_kwargs = {
        'color'     : color,
        'label'     : label,
        **kwargs,
        'markersize': 0,
    }

    plt.plot(
        x_resample,
        y_resample,
        fmt,
        *argslist,
        **line_kwargs
    )

    return x_resample, y_resample
