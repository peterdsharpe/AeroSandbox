import matplotlib.pyplot as plt
import matplotlib as mpl
import aerosandbox.numpy as np
from typing import Tuple, Dict, Union, Callable, List
from scipy import interpolate


def plot_smooth(
        *args,
        color=None,
        label=None,
        function_of: str = None,
        resample_resolution: int = 500,
        drop_nans: bool = False,
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
    ### Parse *args
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
        raise ValueError("Missing plot data. Use syntax `plot_smooth(x, y, fmt, **kwargs)'.")
    else:
        raise ValueError("Unrecognized syntax. Use syntax `plot_smooth(x, y, fmt, **kwargs)'.")

    ### Ensure types are correct (e.g., if a list or Pandas Series is passed in)
    x = np.array(x)
    y = np.array(y)

    if drop_nans:
        nanmask = np.logical_not(
            np.logical_or(
                np.isnan(x),
                np.isnan(y)
            )
        )

        x = x[nanmask]
        y = y[nanmask]

    # At this point, x, y, and fmt are defined.

    ### Resample points

    if function_of is None:
        # Compute the relative spacing of points
        dx = np.diff(x)
        dy = np.diff(y)

        x_rng = np.nanmax(x) - np.nanmin(x)
        y_rng = np.nanmax(y) - np.nanmin(y)

        if x_rng == 0:
            x_rng = 1
        if y_rng == 0:
            y_rng = 1

        dx_norm = dx / x_rng
        dy_norm = dy / y_rng

        ds_norm = np.sqrt(dx_norm ** 2 + dy_norm ** 2)

        s_norm = np.concatenate([
            [0],
            np.nancumsum(ds_norm) / np.nansum(ds_norm)
        ])

        bspline = interpolate.make_interp_spline(
            x=s_norm,
            y=np.stack(
                (x, y), axis=1
            )
        )
        result = bspline(np.linspace(0, 1, resample_resolution))
        x_resample = result[:, 0]
        y_resample = result[:, 1]

    elif function_of == "x":
        x_resample = np.linspace(
            np.nanmin(x),
            np.nanmax(x),
            resample_resolution
        )

        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]

        order = np.argsort(x)

        y_resample = interpolate.PchipInterpolator(
            x=x[order],
            y=y[order],
        )(x_resample)

    elif function_of == "y":

        y_resample = np.linspace(
            np.nanmin(y),
            np.nanmax(y),
            resample_resolution
        )

        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]

        order = np.argsort(y)

        x_resample = interpolate.PchipInterpolator(
            x=y[order],
            y=x[order],
        )(y_resample)

    ### Plot

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

if __name__ == '__main__':
    import aerosandbox.numpy as np

    # t = np.linspace(0, 1, 12)  # Parametric variable
    # x = np.cos(2 * np.pi * t)
    # y = np.cos(2 * np.pi * t ** 4) - t
    #
    # fig, ax = plt.subplots()
    # plot_smooth(
    #     x, y, color='purple'
    # )
    # plt.show()

    fig, ax = plt.subplots()
    x = np.linspace(0, 1, 8)
    plot_smooth(
        x, np.exp(-10 * x**0.5), color='goldenrod',
        function_of="x",
        # markersize=0,
        resample_resolution=2000
    )
    plt.show()