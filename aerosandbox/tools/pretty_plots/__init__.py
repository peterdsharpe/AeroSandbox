"""Provide a set of tools used for making prettier Matplotlib plots."""

### General imports
import seaborn as sns

### Local imports
from aerosandbox.tools.pretty_plots.plots import (
    contour,
    pie,
    plot_color_by_value,
    plot_smooth,
    plot_with_bootstrapped_uncertainty,
)
from aerosandbox.tools.pretty_plots.formatting import show_plot, set_ticks, equal
from aerosandbox.tools.pretty_plots.colors import (
    palettes,
    get_discrete_colors_from_colormap,
    adjust_lightness,
    get_last_line_color,
    mpl,
    plt,
)
from aerosandbox.tools.pretty_plots.annotation import hline, vline
from aerosandbox.tools.pretty_plots.threedim import (
    figure3d,
    ax_is_3d,
    set_preset_3d_view_angle,
)
from aerosandbox.tools.pretty_plots.quickplot import qp

sns.set_theme(
    palette=palettes["categorical"],
)

mpl.rcParams["figure.dpi"] = 200
mpl.rcParams["axes.formatter.useoffset"] = False
mpl.rcParams["contour.negative_linestyle"] = "solid"
