"""
A set of tools used for making prettier Matplotlib plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from typing import Union, Dict, List, Callable, Tuple
from matplotlib import ticker
import aerosandbox.numpy as np
from aerosandbox.tools.string_formatting import eng_string

### Define color palettes
palettes = {
    "categorical": [
        "#4285F4",  # From Google logo, blue
        "#EA4335",  # From Google logo, red
        "#34A853",  # From Google logo, green
        "#ECB22E",  # From Slack logo, gold
        "#9467BD",  # Rest are from Matplotlib "tab10"
        "#8C564B",
        "#E377C2",
        "#7F7F7F",
    ],
}

sns.set_theme(
    palette=palettes["categorical"],
)

mpl.rcParams["figure.dpi"] = 200
mpl.rcParams["axes.formatter.useoffset"] = False
mpl.rcParams["contour.negative_linestyle"] = 'solid'


def set_ticks(
        x_major: Union[float, int] = None,
        x_minor: Union[float, int] = None,
        y_major: Union[float, int] = None,
        y_minor: Union[float, int] = None
):
    ax = plt.gca()
    if x_major is not None:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(base=x_major))
    if x_minor is not None:
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=x_minor))
    if y_major is not None:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(base=y_major))
    if y_minor is not None:
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=y_minor))


def equal():
    plt.gca().set_aspect("equal", adjustable='box')


def figure3d(*args, **kwargs):
    fig = plt.figure(*args, **kwargs)
    ax = plt.axes(projection='3d')
    return fig, ax


def adjust_lightness(color, amount=1.0):
    """
    Converts a color to HLS space, then mulitplies the lightness by `amount`, then converts back to RGB.

    Args:
        color: A color, in any format that matplotlib understands.
        amount: The amount to multiply the lightness by. Valid range is 0 to infinity.

    Returns: A color, as an RGB tuple.

    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def show_plot(
        title: str = None,
        xlabel: str = None,
        ylabel: str = None,
        dpi: float = None,
        savefig: str = None,
        tight_layout: bool = True,
        legend: bool = None,
        legend_frame: bool = True,
        show: bool = True,
        pretty_grids: bool = True,
):
    """
    Finalize and show a plot.
    Args:
        title:
        xlabel:
        ylabel:
        tight_layout:
        legend:
        show:
        pretty_grids:

    Returns:

    """
    fig = plt.gcf()
    axes = fig.get_axes()

    if pretty_grids:
        for ax in axes:
            if not ax.get_label() == '<colorbar>':
                if not ax.name == '3d':
                    ax.grid(True, 'major', axis='both', linewidth=1.6)
                    ax.grid(True, 'minor', axis='both', linewidth=0.7)

    ### Determine if a legend should be shown
    if legend is None:
        lines = plt.gca().lines
        if len(lines) <= 1:
            legend = False
        else:
            legend = False
            for line in lines:
                if line.get_label()[0] != "_":
                    legend = True
                    break

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        if len(axes) <= 1:
            plt.title(title)
        else:
            plt.suptitle(title)
    if tight_layout:
        plt.tight_layout()
    if legend:
        plt.legend(frameon=legend_frame)
    if dpi is not None:
        fig.set_dpi(dpi)
    if savefig is not None:
        plt.savefig(savefig)
    if show:
        plt.show()


def hline(
        y,
        linestyle="--",
        color="k",
        text: str = None,
        text_xloc=0.5,
        text_ha="center",
        text_va="bottom",
        text_kwargs=None,
):  # TODO docs
    if text_kwargs is None:
        text_kwargs = {}
    ax = plt.gca()
    xlim = ax.get_xlim()
    plt.axhline(y=y, ls=linestyle, color=color)
    if text is not None:
        plt.text(
            x=text_xloc * xlim[1] + (1 - text_xloc) * xlim[0],
            y=y,
            s=text,
            color=color,
            horizontalalignment=text_ha,
            verticalalignment=text_va,
            **text_kwargs
        )


def vline(
        x,
        linestyle="--",
        color="k",
        text: str = None,
        text_yloc=0.5,
        text_ha="right",
        text_va="center",
        text_kwargs=None,
):  # TODO docs
    if text_kwargs is None:
        text_kwargs = {}
    ax = plt.gca()
    ylim = ax.get_ylim()
    plt.axvline(x=x, ls=linestyle, color=color)
    if text is not None:
        plt.text(
            x=x,
            y=text_yloc * ylim[1] + (1 - text_yloc) * ylim[0],
            s=text,
            color=color,
            horizontalalignment=text_ha,
            verticalalignment=text_va,
            rotation=90,
            **text_kwargs
        )


def plot_color_by_value(
        x: np.ndarray,
        y: np.ndarray,
        *args,
        c: np.ndarray,
        cmap=mpl.cm.get_cmap('viridis'),
        colorbar=False,
        colorbar_label: str = None,
        clim: Tuple[float, float] = None,
        **kwargs
):
    """
    Uses same syntax as matplotlib.pyplot.plot, except that `c` is now an array-like that maps to a specific color
    pulled from `cmap`. Makes lines that are multicolored based on this `c` value.

    Args:
        x:
        y:
        *args:
        c:
        cmap:
        **kwargs:

    Returns:

    """
    cmap = mpl.cm.get_cmap(cmap)

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
            color=cmap(norm((c1 + c2) / 2) if cmin != cmax else 0.5),
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
