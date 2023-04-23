import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from typing import Tuple, Dict, Union, Callable, List
from scipy import interpolate
from aerosandbox.tools.string_formatting import eng_string


def contour(
        *args,
        levels: Union[int, List, np.ndarray] = 31,
        colorbar: bool = True,
        linelabels: bool = True,
        cmap=None,
        alpha: float = 0.7,
        extend: str = "neither",
        linecolor="k",
        linewidths: float = 0.5,
        extendrect: bool = True,
        linelabels_format: Union[str, Callable[[float], str]] = eng_string,
        linelabels_fontsize: float = 8,
        max_side_length_nondim: float = np.Inf,
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
    if X is None:
        X = np.arange(Z.shape[1])
    if Y is None:
        Y = np.arange(Z.shape[0])

    ### Set defaults
    if cmap is None:
        cmap = mpl.colormaps.get_cmap('viridis')
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

        Z_ratio = np.nanmax(Z) / np.nanmin(Z)
        log10_ceil_z_max = np.ceil(np.log10(np.nanmax(Z)))
        log10_floor_z_min = np.floor(np.log10(np.nanmin(Z)))

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
            "norm": mpl.colors.LogNorm(),
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

    try:  ### If this works, then the data is gridded (i.e. X and Y are 2D arrays)
        cont = plt.contour(X, Y, Z, **contour_kwargs)
        contf = plt.contourf(X, Y, Z, **contourf_kwargs)

    except TypeError as e:  ### If this fails, then the data is unstructured (i.e. X and Y are 1D arrays)

        ### Drop all NaN values
        mask = np.logical_not(
            np.logical_or.reduce(
                [np.isnan(X), np.isnan(Y), np.isnan(Z)]
            )
        )
        X = X[mask]
        Y = Y[mask]
        Z = Z[mask]

        ### Create the triangulation
        tri = mpl.tri.Triangulation(X, Y)

        ### Filter out extrapolation that's too large
        # See also: https://stackoverflow.com/questions/42426095/matplotlib-contour-contourf-of-concave-non-gridded-data
        X_scale = np.nanmax(X) - np.nanmin(X)
        Y_scale = np.nanmax(Y) - np.nanmin(Y)

        triangles = tri.triangles
        xtri = X[triangles] - np.roll(X[triangles], 1, axis=1)
        ytri = Y[triangles] - np.roll(Y[triangles], 1, axis=1)
        maxi = np.max(
            np.sqrt(
                (xtri / X_scale) ** 2 +
                (ytri / Y_scale) ** 2
            ),
            axis=1
        )

        tri.set_mask(maxi > max_side_length_nondim)

        try:

            cont = plt.tricontour(tri, Z, **contour_kwargs)
            contf = plt.tricontourf(tri, Z, **contourf_kwargs)
        except TypeError:
            raise e

    if colorbar:
        from matplotlib import cm

        cbar = plt.colorbar(
            ax=contf.axes,
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

    cmap = plt.get_cmap("rainbow")

    cont, contf, cbar = contour(
        X, Y, Z,
        z_log_scale=True,
        cmap=cmap,
        levels=20,
        colorbar_label="Colorbar label"
    )
    p.show_plot(
        "Title",
        "X label",
        "Y label"
    )
