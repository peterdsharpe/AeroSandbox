import matplotlib.pyplot as plt
from matplotlib import ticker
from typing import Union
from .labellines import labelLines


def show_plot(
        title: str = None,
        xlabel: str = None,
        ylabel: str = None,
        dpi: float = None,
        savefig: str = None,
        tight_layout: bool = True,
        legend: bool = None,
        legend_inline: bool = False,
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
        if legend_inline:  # Display an inline (matplotlib-label-lines) legend instead
            labelLines(
                lines=plt.gca().get_lines(),
            )
        else:  # Display a traditional legend
            plt.legend(frameon=legend_frame)
    if dpi is not None:
        fig.set_dpi(dpi)
    if savefig is not None:
        plt.savefig(savefig)
    if show:
        plt.show()


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
