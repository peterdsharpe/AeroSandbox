import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from typing import Union
from matplotlib import ticker

# plt.ion()

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


def adjust_lightness(color, amount=0.5):
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
    if show:
        plt.show()

# def contour(
#         func: Callable,
#         x_range: Tuple[Union[float,int],Union[float,int]],
#         y_range: Tuple[Union[float,int],Union[float,int]],
#         resolution:int =50,
#         show:bool=True,  # type: bool
# ):
#     """
#     Makes a contour plot of a function of 2 variables. Can also plot a list of functions.
#     :param func: function of form f(x,y) to plot.
#     :param x_range: Range of x values to plot, expressed as a tuple (x_min, x_max)
#     :param y_range: Range of y values to plot, expressed as a tuple (y_min, y_max)
#     :param resolution: Resolution in x and y to plot. [int]
#     :param show: Should we show the plot?
#     :return:
#     """
#     fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
#     x = np.linspace(x_range[0], x_range[1], resolution)
#     y = np.linspace(y_range[0], y_range[1], resolution)
#     # TODO finish function
