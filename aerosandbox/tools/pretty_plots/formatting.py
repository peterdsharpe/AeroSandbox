import matplotlib.pyplot as plt
from matplotlib import ticker as mt
from typing import Union
from aerosandbox.tools.pretty_plots.labellines import labelLines
import aerosandbox.numpy as np
from aerosandbox.tools.pretty_plots.threedim import ax_is_3d
from functools import partial


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
        set_ticks: bool = True,
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
                    for i_ax in [ax.xaxis, ax.yaxis, ax.zaxis]:
                        i_ax._axinfo["grid"].update(dict(
                            linewidth=0.7,
                        ))
                        i_ax.set_tick_params(which="minor", color=(0, 0, 0, 0))

    if set_ticks:

        for ax in axes:

            individual_axes_and_limits = {
                ax.xaxis: ax.get_xlim(),
                ax.yaxis: ax.get_ylim(),
            }
            if hasattr(ax, "zaxis"):
                individual_axes_and_limits[ax.zaxis] = ax.get_zlim()

            for i_ax, lims in individual_axes_and_limits.items():

                maj_loc = None
                maj_fmt = None
                min_loc = None
                min_fmt = None

                if i_ax.get_scale() == "log":

                    def linlogfmt(x, pos, ticks=[1.], default="", base=10):
                        if x < 0:
                            sign_string = "-"
                            x = -x
                        else:
                            sign_string = ""

                        exponent = np.floor(np.log(x) / np.log(base))
                        coeff = x / base ** exponent

                        ### Fix any floating-point error during the floor function
                        if coeff < 1:
                            coeff *= base
                            exponent -= 1
                        elif coeff >= base:
                            coeff /= base
                            exponent += 1

                        for tick in ticks:
                            if np.isclose(coeff, tick):
                                return r"$\mathdefault{%s%g}$" % (
                                    sign_string,
                                    x
                                )

                        return default

                    def logfmt(x, pos, ticks=[1.], default="", base=10):
                        if x < 0:
                            sign_string = "-"
                            x = -x
                        else:
                            sign_string = ""

                        exponent = np.floor(np.log(x) / np.log(base))
                        coeff = x / base ** exponent

                        ### Fix any floating-point error during the floor function
                        if coeff < 1:
                            coeff *= base
                            exponent -= 1
                        elif coeff >= base:
                            coeff /= base
                            exponent += 1

                        for tick in ticks:
                            if tick == 1:
                                if np.isclose(coeff, 1):
                                    return r"$\mathdefault{%s%s^{%d}}$" % (
                                        sign_string,
                                        base,
                                        exponent
                                    )
                            else:
                                if np.isclose(coeff, tick):
                                    # return f"${base:.0f} {{\\times 10^{int(exponent)}}}$"
                                    return r"$\mathdefault{%s%g\times%s^{%d}}$" % (
                                        sign_string,
                                        coeff,
                                        base,
                                        exponent
                                    )
                        return default

                    ratio = lims[1] / lims[0]

                    i_ax.set_tick_params(which="minor", labelsize=8)

                    if ratio < 10:
                        maj_loc = mt.MaxNLocator(
                            nbins=6,
                            steps=[1, 2, 5, 10],
                            min_n_ticks=4,
                        )
                        maj_fmt = mt.ScalarFormatter()

                        class LogAutoMinorLocator(mt.AutoMinorLocator):
                            """
                            Dynamically find minor tick positions based on the positions of
                            major ticks. The scale must be linear with major ticks evenly spaced.
                            """

                            def __call__(self):
                                majorlocs = self.axis.get_majorticklocs()
                                try:
                                    majorstep = majorlocs[1] - majorlocs[0]
                                except IndexError:
                                    # Need at least two major ticks to find minor tick locations
                                    # TODO: Figure out a way to still be able to display minor
                                    # ticks without two major ticks visible. For now, just display
                                    # no ticks at all.
                                    return []

                                if self.ndivs is None:

                                    majorstep_no_exponent = 10 ** (np.log10(majorstep) % 1)

                                    if np.isclose(majorstep_no_exponent, [1.0, 2.5, 5.0, 10.0]).any():
                                        ndivs = 5
                                    else:
                                        ndivs = 4
                                else:
                                    ndivs = self.ndivs

                                minorstep = majorstep / ndivs

                                vmin, vmax = self.axis.get_view_interval()
                                if vmin > vmax:
                                    vmin, vmax = vmax, vmin

                                t0 = majorlocs[0]
                                tmin = ((vmin - t0) // minorstep + 1) * minorstep
                                tmax = ((vmax - t0) // minorstep + 1) * minorstep
                                locs = np.arange(tmin, tmax, minorstep) + t0

                                return self.raise_if_exceeds(locs)

                        min_loc = LogAutoMinorLocator()
                        min_fmt = mt.NullFormatter()

                    elif ratio < 10 ** 1.5:
                        maj_loc = mt.LogLocator(subs=np.arange(1, 10))
                        maj_fmt = mt.FuncFormatter(
                            partial(linlogfmt, ticks=[1, 2, 5], default=r"$^{^|}$")
                        )
                        min_loc = mt.LogLocator(numticks=999, subs=np.arange(10, 100) / 10)
                        min_fmt = mt.NullFormatter()
                    elif ratio < 10 ** 2.5:
                        maj_loc = mt.LogLocator()
                        maj_fmt = mt.FuncFormatter(partial(logfmt, ticks=[1]))
                        min_loc = mt.LogLocator(numticks=999, subs=np.arange(1, 10))
                        min_fmt = mt.FuncFormatter(partial(logfmt, ticks=[2, 5]))
                    elif ratio < 10 ** 8:
                        maj_loc = mt.LogLocator()
                        maj_fmt = mt.FuncFormatter(partial(logfmt, ticks=[1]))
                        min_loc = mt.LogLocator(numticks=999, subs=np.arange(1, 10))
                        min_fmt = mt.FuncFormatter(partial(logfmt, ticks=[1]))
                    elif ratio < 10 ** 16:
                        maj_loc = mt.LogLocator()
                        maj_fmt = mt.LogFormatterSciNotation()
                        min_loc = mt.LogLocator(numticks=999, subs=np.arange(1, 10))
                        min_fmt = mt.NullFormatter()
                    else:
                        pass

                elif i_ax.get_scale() == "linear":
                    maj_loc = mt.MaxNLocator(
                        nbins='auto',
                        steps=[1, 2, 5, 10],
                        min_n_ticks=3,
                    )

                    min_loc = mt.AutoMinorLocator()

                else: # For any other scale, just use the default tick locations
                    continue

                if len(i_ax.get_major_ticks()) != 0:  # Unless the user has manually set the ticks to be empty
                    if maj_loc is not None:
                        i_ax.set_major_locator(maj_loc)
                    if min_loc is not None:
                        i_ax.set_minor_locator(min_loc)

                if maj_fmt is not None:
                    i_ax.set_major_formatter(maj_fmt)
                if min_fmt is not None:
                    i_ax.set_minor_formatter(min_fmt)

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
        ax.xaxis.set_major_locator(mt.MultipleLocator(base=x_major))
    if x_minor is not None:
        ax.xaxis.set_minor_locator(mt.MultipleLocator(base=x_minor))
    if y_major is not None:
        ax.yaxis.set_major_locator(mt.MultipleLocator(base=y_major))
    if y_minor is not None:
        ax.yaxis.set_minor_locator(mt.MultipleLocator(base=y_minor))
    if z_major is not None:
        ax.zaxis.set_major_locator(mt.MultipleLocator(base=z_major))
    if z_minor is not None:
        ax.zaxis.set_minor_locator(mt.MultipleLocator(base=z_minor))


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
