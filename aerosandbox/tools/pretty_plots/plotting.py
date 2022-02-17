import matplotlib.pyplot as plt
import matplotlib as mpl
import aerosandbox.numpy as np
from typing import Tuple, Dict, Union, Callable, List
from aerosandbox.tools.string_formatting import eng_string


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


def contour(
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        levels: Union[int, List, np.ndarray] = 31,
        colorbar: bool = True,
        linelabels: bool = True,
        cmap=mpl.cm.get_cmap('viridis'),
        alpha: float = 0.7,
        extend: str = "both",
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

    if contour_kwargs is None:
        contour_kwargs = {}
    if contourf_kwargs is None:
        contourf_kwargs = {}
    if colorbar_kwargs is None:
        colorbar_kwargs = {}
    if linelabels_kwargs is None:
        linelabels_kwargs = {}

    args = [
        X,
        Y,
        Z,
    ]

    shared_kwargs = kwargs
    if levels is not None:
        shared_kwargs["levels"] = levels
    if alpha is not None:
        shared_kwargs["alpha"] = alpha
    if extend is not None:
        shared_kwargs["extend"] = extend
    if z_log_scale:

        shared_kwargs = {
            "norm"   : mpl.colors.LogNorm(),
            "locator": mpl.ticker.LogLocator(
                subs=np.geomspace(1, 10, 4 + 1)[:-1]
            ),
            **shared_kwargs
        }

        if np.min(Z) <= 0:
            import warnings
            warnings.warn(
                "Warning: All values of the `Z` input to `contour()` should be nonnegative if `z_log_scale` is True!",
                stacklevel=2
            )
            Z = np.maximum(Z, 1e-300)  # Make all values nonnegative

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

    try:
        cont = plt.contour(*args, **contour_kwargs)
        contf = plt.contourf(*args, **contourf_kwargs)
    except TypeError as e:
        try:
            cont = plt.tricontour(*args, **contour_kwargs)
            contf = plt.tricontourf(*args, **contourf_kwargs)
        except TypeError:
            raise e

    if colorbar:
        cbar = plt.colorbar(**colorbar_kwargs)

        if z_log_scale:
            cbar.ax.yaxis.set_major_locator(mpl.ticker.LogLocator())
            cbar.ax.yaxis.set_major_formatter(mpl.ticker.LogFormatter())
    else:
        cbar = None

    if linelabels:
        plt.gca().clabel(cont, **linelabels_kwargs)

    return cont, contf, cbar
