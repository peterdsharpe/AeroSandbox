import matplotlib.pyplot as plt
import matplotlib as mpl
import aerosandbox.numpy as np
from typing import Tuple, Dict, Union, Callable, List
from aerosandbox.tools.string_formatting import eng_string
from scipy import interpolate


def plot_color_by_value(
        x: np.ndarray,
        y: np.ndarray,
        *args,
        c: np.ndarray,
        cmap=mpl.cm.get_cmap('turbo'),
        colorbar: bool = False,
        colorbar_label: str = None,
        clim: Tuple[float, float] = None,
        **kwargs
):
    """
    Uses same syntax as matplotlib.pyplot.plot, except that `c` is now an array-like that maps to a specific color
    pulled from `cmap`. Makes lines that are multicolored based on this `c` value.

    Args:

        x: Array of x-points.

        y: Array of y-points.

        *args: Args that will be passed into matplotlib.pyplot.plot().
            Example: ".-" for a dotted line.

        c: Array of values that will map to colors. Must be the same length as x and y.

        cmap: The colormap to use.

        colorbar: Whether or not to display the colormap. [bool]

        colorbar_label: The label to add to the colorbar. Only applies if the colorbar is created. [str]

        clim: A tuple of (min, max) that assigns bounds to the colormap. Computed from the range of `c` if not given.

        **kwargs: Kwargs that will be passed into matplotlib.pyplot.plot()


    Returns:

    """
    cmap = mpl.cm.get_cmap(cmap)

    x = np.array(x)
    y = np.array(y)
    c = np.array(c)

    cmin = c.min()
    cmax = c.max()

    if clim is None:
        clim = (cmin, cmax)

    norm = plt.Normalize(vmin=clim[0], vmax=clim[1], clip=False)

    label = kwargs.pop("label", None)

    lines = []

    for i, (
            x1, x2,
            y1, y2,
            c1, c2,
    ) in enumerate(zip(
        x[:-1], x[1:],
        y[:-1], y[1:],
        c[:-1], c[1:],
    )):
        line = plt.plot(
            [x1, x2],
            [y1, y2],
            *args,
            color=cmap(
                norm(
                    (c1 + c2) / 2
                ) if cmin != cmax else 0.5
            ),
            **kwargs
        )
        lines += line

    if label is not None:
        line = plt.plot(
            [None],
            [None],
            *args,
            color=cmap(0.5),
            label=label,
            **kwargs
        )
        lines += line
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    if colorbar:
        if colorbar_label is None:
            cbar = plt.colorbar(sm)
        else:
            cbar = plt.colorbar(sm, label=colorbar_label)
    else:
        cbar = None
    return lines, sm, cbar


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


def contour(
        *args,
        levels: Union[int, List, np.ndarray] = 31,
        colorbar: bool = True,
        linelabels: bool = True,
        cmap=mpl.cm.get_cmap('viridis'),
        alpha: float = 0.7,
        extend: str = "neither",
        linecolor="k",
        linewidths: float = 0.5,
        extendrect: bool = True,
        linelabels_format: Union[str, Callable[[float], str]] = eng_string,
        linelabels_fontsize: float = 8,
        colorbar_label: str = None,
        z_log_scale: bool = False,
        contour_kwargs: Dict = None,
        contourf_kwargs: Dict = None,
        colorbar_kwargs: Dict = None,
        linelabels_kwargs: Dict = None,
        **kwargs,
):
    """
    An analogue for plt.contour and plt.tricontour and friends that produces a much prettier default graph.

    Can take inputs with either contour or tricontour syntax.

    See syntax here:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contour.html
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contourf.html
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.tricontour.html
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.tricontourf.html

    Args:
        X: See contour docs.

        Y: See contour docs.

        Z: See contour docs.

        levels: See contour docs.

        colorbar: Should we draw a colorbar?

        linelabels: Should we add line labels?

        cmap: What colormap should we use?

        alpha: What transparency should all plot elements be?

        extend: See contour docs.

        linecolor: What color should the line labels be?

        linewidths: See contour docs.

        extendrect: See colorbar docs.

        linelabels_format: See ax.clabel docs.

        linelabels_fontsize: See ax.clabel docs.

        contour_kwargs: Additional keyword arguments for contour.

        contourf_kwargs: Additional keyword arguments for contourf.

        colorbar_kwargs: Additional keyword arguments for colorbar.

        linelabels_kwargs: Additional keyword arguments for the line labels (ax.clabel).

        **kwargs: Additional keywords, which are passed to both contour and contourf.


    Returns: A tuple of (contour, contourf, colorbar) objects.

    """
    bad_signature_error = ValueError("Call signature should be one of:\n"
                                     "  * `contour(Z, **kwargs)`\n"
                                     "  * `contour(X, Y, Z, **kwargs)`\n"
                                     "  * `contour(X, Y, Z, levels, **kwargs)`"
                                     )

    ### Parse *args
    if len(args) == 1:
        X = None
        Y = None
        Z = args[0]
    elif len(args) == 3:
        X = args[0]
        Y = args[1]
        Z = args[2]
    else:
        raise bad_signature_error
    if not len(Z.shape) == 2:
        raise ValueError(f"Z must be a 2D array; instead, got shape {Z.shape}.")
    if X is None:
        X = np.arange(Z.shape[1])
    if Y is None:
        Y = np.arange(Z.shape[0])

    ### Set defaults
    if contour_kwargs is None:
        contour_kwargs = {}
    if contourf_kwargs is None:
        contourf_kwargs = {}
    if colorbar_kwargs is None:
        colorbar_kwargs = {}
    if linelabels_kwargs is None:
        linelabels_kwargs = {}

    shared_kwargs = kwargs
    if levels is not None:
        shared_kwargs["levels"] = levels
    if alpha is not None:
        shared_kwargs["alpha"] = alpha
    if extend is not None:
        shared_kwargs["extend"] = extend
    if z_log_scale:
        if np.any(Z) <= 0:
            raise ValueError(
                "All values of the `Z` input to `contour()` should be nonnegative if `z_log_scale` is True!"
            )

        Z_ratio = np.max(Z) / np.min(Z)
        log10_ceil_z_max = np.ceil(np.log10(np.max(Z)))
        log10_floor_z_min = np.floor(np.log10(np.min(Z)))

        try:
            default_levels = int(levels)
        except TypeError:
            default_levels = 31
        divisions_per_decade = np.ceil(default_levels / np.log10(Z_ratio)).astype(int)

        if Z_ratio > 1e8:
            locator = mpl.ticker.LogLocator()
        else:
            locator = mpl.ticker.LogLocator(
                subs=np.geomspace(1, 10, divisions_per_decade + 1)[:-1]
            )

        shared_kwargs = {
            "norm"   : mpl.colors.LogNorm(),
            "locator": locator,
            **shared_kwargs
        }

        colorbar_kwargs = {
            "norm" : mpl.colors.LogNorm(),
            **colorbar_kwargs
        }

    if colorbar_label is not None:
        colorbar_kwargs["label"] = colorbar_label

    contour_kwargs = {
        "colors"    : linecolor,
        "linewidths": linewidths,
        **shared_kwargs,
        **contour_kwargs
    }
    contourf_kwargs = {
        "cmap": cmap,
        **shared_kwargs,
        **contourf_kwargs
    }

    colorbar_kwargs = {
        "extendrect": extendrect,
        **colorbar_kwargs
    }

    linelabels_kwargs = {
        "inline"  : 1,
        "fontsize": linelabels_fontsize,
        "fmt"     : linelabels_format,
        **linelabels_kwargs
    }

    ### Now, with all the kwargs merged, do the actual plotting:

    try:
        cont = plt.contour(X, Y, Z, **contour_kwargs)
        contf = plt.contourf(X, Y, Z, **contourf_kwargs)
    except TypeError as e:
        try:
            cont = plt.tricontour(X, Y, Z, **contour_kwargs)
            contf = plt.tricontourf(X, Y, Z, **contourf_kwargs)
        except TypeError:
            raise e

    if colorbar:
        from matplotlib import cm

        cbar = plt.colorbar(
            mappable=cm.ScalarMappable(
                norm=contf.norm,
                cmap=contf.cmap,
            ),
            **colorbar_kwargs
        )

        if z_log_scale:

            if Z_ratio >= 1e2:
                cbar.ax.yaxis.set_major_locator(mpl.ticker.LogLocator())
                cbar.ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(subs=np.arange(10)))
                cbar.ax.yaxis.set_major_formatter(mpl.ticker.LogFormatterSciNotation())
                cbar.ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
            elif Z_ratio >= 10 ** 1.5:
                cbar.ax.yaxis.set_major_locator(mpl.ticker.LogLocator())
                cbar.ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(subs=np.arange(10)))
                cbar.ax.yaxis.set_major_formatter(mpl.ticker.LogFormatterSciNotation())
                cbar.ax.yaxis.set_minor_formatter(mpl.ticker.LogFormatterSciNotation())
            else:
                cbar.ax.yaxis.set_major_locator(mpl.ticker.LogLocator(subs=np.arange(10)))
                cbar.ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(subs=np.arange(100)))
                cbar.ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
                cbar.ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    else:
        cbar = None

    if linelabels:
        cont.axes.clabel(cont, **linelabels_kwargs)

    return cont, contf, cbar


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    x = np.linspace(0, 1, 300)
    y = np.linspace(0, 1, 200)
    X, Y = np.meshgrid(x, y)

    Z_ratio = 1.5

    Z = 10 ** (
            Z_ratio / 2 * np.cos(
        2 * np.pi * (X ** 4 + Y ** 4)
    )
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    try:
        from palettable.scientific.sequential import Hawaii_20

        cmap = Hawaii_20.get_mpl_colormap()
    except ImportError:
        cmap = plt.get_cmap("rainbow")

    cont, contf, cbar = contour(
        X, Y, Z,
        z_log_scale=True,
        cmap=cmap,
        levels=10,
        colorbar_label="Colorbar label"
    )
    p.show_plot(
        "Title",
        "X label",
        "Y label"
    )
