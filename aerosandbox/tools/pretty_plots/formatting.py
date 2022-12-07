import matplotlib.pyplot as plt
from matplotlib import ticker
from typing import Union
from .labellines import labelLines
import aerosandbox.numpy as np
from .threedim import ax_is_3d


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
                if not ax_is_3d(ax):
                    if any(line.get_visible() for line in ax.get_xgridlines()):
                        ax.grid(True, 'major', axis='x', linewidth=1.6)
                        ax.grid(True, 'minor', axis='x', linewidth=0.7)
                    if any(line.get_visible() for line in ax.get_ygridlines()):
                        ax.grid(True, 'major', axis='y', linewidth=1.6)
                        ax.grid(True, 'minor', axis='y', linewidth=0.7)
                else:
                    pass  # TODO
                    # if any(line.get_visible() for line in ax.get_xgridlines()):
                    #     ax.grid(True, 'major', axis='x', linewidth=1.6)
                    #     ax.grid(True, 'minor', axis='x', linewidth=0.7)
                    # if any(line.get_visible() for line in ax.get_ygridlines()):
                    #     ax.grid(True, 'major', axis='y', linewidth=1.6)
                    #     ax.grid(True, 'minor', axis='y', linewidth=0.7)
                    # if any(line.get_visible() for line in ax.get_zgridlines()):
                    #     ax.grid(True, 'major', axis='z', linewidth=1.6)
                    #     ax.grid(True, 'minor', axis='z', linewidth=0.7)

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
        y_minor: Union[float, int] = None,
        z_major: Union[float, int] = None,
        z_minor: Union[float, int] = None,
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
    if z_major is not None:
        ax.zaxis.set_major_locator(ticker.MultipleLocator(base=z_major))
    if z_minor is not None:
        ax.zaxis.set_minor_locator(ticker.MultipleLocator(base=z_minor))


def equal() -> None:
    """
    Sets all axes to be equal. Works for both 2d plots and 3d plots.

    Returns: None

    """
    ax = plt.gca()

    if not ax_is_3d(ax):
        ax.set_aspect("equal", adjustable='box')

    else:
        ax.set_box_aspect((1, 1, 1))

        xlim = ax.get_xlim3d()
        ylim = ax.get_ylim3d()
        zlim = ax.get_zlim3d()

        x_range = abs(xlim[1] - xlim[0])
        x_middle = np.mean(xlim)
        y_range = abs(ylim[1] - ylim[0])
        y_middle = np.mean(ylim)
        z_range = abs(zlim[1] - zlim[0])
        z_middle = np.mean(zlim)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
