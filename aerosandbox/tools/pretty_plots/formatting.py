import matplotlib.pyplot as plt
from matplotlib import ticker as mt
from typing import Union, List
from aerosandbox.tools.pretty_plots.labellines import labelLines
import aerosandbox.numpy as np
from aerosandbox.tools.pretty_plots.threedim import ax_is_3d
from functools import partial
from pathlib import Path
from aerosandbox.tools import string_formatting as sf


def show_plot(
        title: str = None,
        xlabel: str = None,
        ylabel: str = None,
        zlabel: str = None,
        dpi: float = None,
        savefig: Union[str, List[str]] = None,
        savefig_transparent: bool = False,
        tight_layout: bool = True,
        legend: bool = None,
        legend_inline: bool = False,
        legend_frame: bool = True,
        pretty_grids: bool = True,
        set_ticks: bool = True,
        rotate_axis_labels: bool = True,
        rotate_axis_labels_linewidth: int = 14,
        show: bool = True,
):
    """
    Makes a matplotlib Figure (and all its constituent Axes) look "nice", then displays it.

    Arguments control whether various changes (from the default matplotlib settings) are made to the plot.

    One argument in particular, `show` (a boolean), controls whether the plot is displayed.

    Args:

        title: If given, sets the title of the plot. If the Figure has multiple axes, this sets the Figure-level
            suptitle instead of setting the individual Axis title.

        xlabel: If given, sets the xlabel of all axes. (Equivalent to `ax.set_xlabel(my_label)`)

        ylabel: If given, sets the ylabel of all axes. (Equivalent to `ax.set_ylabel(my_label)`)

        zlabel: If given, sets the zlabel of all axes, if the axis is 3D. (Equivalent to `ax.set_zlabel(my_label)`)

        dpi: If given, sets the dpi (display resolution, in Dots Per Inch) of the Figure.

        savefig: If given, saves the figure to the given path(s).

            * If a string is given, saves the figure to that path.
                (E.g., `savefig="my_plot.png"`)

            * If a list of strings is given, saves the figure to each of those paths.
                (E.g., `savefig=["my_plot.png", "my_plot.pdf"]`)

        savefig_transparent: If True, saves the figure with a transparent background. If False, saves the figure with
            a white background. Only has an effect if `savefig` is not None.

        tight_layout: If True, calls `plt.tight_layout()` to adjust the spacing of individual Axes. If False, skips
            this step.

        legend: This value can be True, False, or None.

            * If True, displays a legend on the current Axis.

            * If False, does not add a legend on the current Axis. (However, does not delete any existing legends.)

            * If None (default), goes through some logic to determine whether a legend should be displayed. If there
                is only one line on the current Axis, no legend is displayed. If there are multiple lines, a legend is (
                in general) displayed.

        legend_inline: Boolean that controls whether an "inline" legend is displayed.

            * If True, displays an "inline" legend, where the labels are next to the lines instead of in a box.

            * If False (default), displays a traditional legend.

            Only has an effect if `legend=True` (or `legend=None`, and logic determines that a legend should be
            displayed).

        legend_frame: Boolean that controls whether a frame (rectangular background box) is displayed around the legend.
            Default is True.

        pretty_grids: Boolean that controls whether the gridlines are formatted have linewidths that are (subjectively)
            more readable.

        set_ticks: Boolean that controls whether the tick and grid locations + labels are formatted to be (
            subjectively) more readable.

            Works with both linear and log scales, and with both 2D and 3D plots.

        show: Boolean that controls whether the plot is displayed after all plot changes are applied. Default is
            True. You may want to set this to False if you want to make additional manual changes to the plot before
            displaying it. Default is True.

    Returns: None (completely in-place function). If `show=True` (default), displays the plot after applying changes.

    """
    fig = plt.gcf()
    axes = fig.get_axes()
    axes_with_3D = [ax for ax in axes if ax_is_3d(ax)]

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

                    def linlogfmt(x, pos, ticks=None, default="", base=10):
                        if ticks is None:
                            ticks = [1.]

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
                        # if i_ax.axis_name == "x":
                        #     default = r"$^{^|}$"
                        # elif i_ax.axis_name == "y":
                        #     default = r"–"
                        # else:
                        #     default = ""

                        maj_fmt = mt.FuncFormatter(
                            partial(linlogfmt, ticks=[1, 2, 5])
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

                else:  # For any other scale, just use the default tick locations
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
        legend = False

        for ax in axes:
            lines = ax.get_lines()
            if len(lines) > 1:
                if not all(line.get_label()[0] == "_" for line in lines):
                    legend = True
                    break

    # Make axis labels if needed
    if xlabel is not None:
        for ax in axes:
            if not ax.get_label() == '<colorbar>':
                ax.set_xlabel(xlabel)
    if ylabel is not None:
        for ax in axes:
            if not ax.get_label() == '<colorbar>':
                ax.set_ylabel(ylabel)
    if zlabel is not None:
        if len(axes_with_3D) == 0:
            import warnings
            warnings.warn(
                "You specified a `zlabel`, but there are no 3D axes in this figure. Ignoring `zlabel`.",
                stacklevel=2
            )

        for ax in axes_with_3D:
            if not ax.get_label() == '<colorbar>':
                ax.set_zlabel(zlabel)

    # Rotate axis labels if needed
    if rotate_axis_labels:
        for ax in axes:
            if not ax_is_3d(ax):
                if not ax.get_label() == '<colorbar>':

                    ylabel = ax.get_ylabel()

                    if (rotate_axis_labels_linewidth is not None) and ("\n" not in ylabel):
                        ylabel = sf.wrap_text_ignoring_mathtext(
                            ylabel,
                            width=rotate_axis_labels_linewidth,
                        )

                    ax.set_ylabel(
                        ylabel,
                        rotation=0,
                        ha="right",
                        va="center",
                    )

    if title is not None:
        if len(axes) > 1:
            plt.suptitle(title)
        else:
            plt.title(title)

    if tight_layout:
        plt.tight_layout()

    if legend:
        if legend_inline:  # Display an inline (matplotlib-label-lines) legend instead
            for ax in axes:
                labelLines(
                    lines=ax.get_lines(),
                )
        else:  # Display a traditional legend on the last axis
            plt.legend(frameon=legend_frame)
    if dpi is not None:
        fig.set_dpi(dpi)
    if savefig is not None:

        if not isinstance(savefig, (list, tuple, set)):
            savefig: List[Union[str, Path]] = [savefig]

        for savefig_i in savefig:
            plt.savefig(savefig_i, transparent=savefig_transparent)

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
