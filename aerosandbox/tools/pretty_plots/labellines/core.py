import warnings
from math import atan2, degrees

import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.container import ErrorbarContainer
from matplotlib.dates import DateConverter, num2date

from aerosandbox.tools.pretty_plots.labellines.utils import ensure_float, maximum_bipartite_matching, always_iterable


# Label line with line2D label data
def labelLine(
        line,
        x,
        label=None,
        align=True,
        drop_label=False,
        yoffset=0,
        yoffset_logspace=False,
        outline_color="auto",
        outline_width=8,
        **kwargs,
):
    """Label a single matplotlib line at position x

    Parameters
    ----------
    line : matplotlib.lines.Line
       The line holding the label
    x : number
       The location in data unit of the label
    label : string, optional
       The label to set. This is inferred from the line by default
    drop_label : bool, optional
       If True, the label is consumed by the function so that subsequent
       calls to e.g. legend do not use it anymore.
    yoffset : double, optional
        Space to add to label's y position
    yoffset_logspace : bool, optional
        If True, then yoffset will be added to the label's y position in
        log10 space
    outline_color : None | "auto" | color
        Colour of the outline. If set to "auto", use the background color.
        If set to None, do not draw an outline.
    outline_width : number
        Width of the outline
    kwargs : dict, optional
       Optional arguments passed to ax.text
    """

    ax = line.axes
    xdata = ensure_float(line.get_xdata())
    ydata = line.get_ydata()

    mask = np.isfinite(ydata)
    if mask.sum() == 0:
        raise Exception(f"The line {line} only contains nan!")

    # Find first segment of xdata containing x
    if isinstance(xdata, tuple) and len(xdata) == 2:
        i = 0
        xa = min(xdata)
        xb = max(xdata)
    else:
        for imatch, (xa, xb) in enumerate(zip(xdata[:-1], xdata[1:])):
            if min(xa, xb) <= ensure_float(x) <= max(xa, xb):
                i = imatch
                break
        else:
            raise Exception("x label location is outside data range!")

    xfa = ensure_float(xa)
    xfb = ensure_float(xb)
    ya = ydata[i]
    yb = ydata[i + 1]

    # Handle vertical case
    if xfb == xfa:
        fraction = 0.5
    else:
        fraction = (ensure_float(x) - xfa) / (xfb - xfa)

    if yoffset_logspace:
        y = ya + (yb - ya) * fraction
        y *= 10 ** yoffset
    else:
        y = ya + (yb - ya) * fraction + yoffset

    if not (np.isfinite(ya) and np.isfinite(yb)):
        warnings.warn(
            (
                "%s could not be annotated due to `nans` values. "
                "Consider using another location via the `x` argument."
            )
            % line,
            UserWarning,
        )
        return

    if not label:
        label = line.get_label()

    if drop_label:
        line.set_label(None)

    if align:
        if ax.get_aspect() == "auto":
            # Compute the slope and label rotation
            screen_dx, screen_dy = (
                    ax.transData.transform((xfb, yb)) -
                    ax.transData.transform((xfa, ya))
            )
        elif isinstance(ax.get_aspect(), (float, int)):
            screen_dx = xfb - xfa
            screen_dy = (yb - ya) * ax.get_aspect()
        rotation = (degrees(atan2(screen_dy, screen_dx)) + 90) % 180 - 90
    else:
        rotation = 0

    # Set a bunch of keyword arguments
    if "color" not in kwargs:
        kwargs["color"] = line.get_color()

    if ("horizontalalignment" not in kwargs) and ("ha" not in kwargs):
        kwargs["ha"] = "center"

    if ("verticalalignment" not in kwargs) and ("va" not in kwargs):
        kwargs["va"] = "center"

    if "clip_on" not in kwargs:
        kwargs["clip_on"] = True

    if "zorder" not in kwargs:
        kwargs["zorder"] = 2.5

    if outline_color == "auto":
        outline_color = ax.get_facecolor()

    txt = ax.text(x, y, label, rotation=rotation, **kwargs)

    if outline_color is None:
        effects = [path_effects.Normal()]
    else:
        effects = [
            path_effects.Stroke(linewidth=outline_width, foreground=outline_color),
            path_effects.Normal(),
        ]

    txt.set_path_effects(effects)
    return txt


def labelLines(
        lines,
        align=True,
        xvals=None,
        drop_label=False,
        shrink_factor=0.05,
        yoffsets=0,
        outline_color="auto",
        outline_width=5,
        **kwargs,
):
    """Label all lines with their respective legends.

    Parameters
    ----------
    lines : list of matplotlib lines
       The lines to label
    align : boolean, optional
       If True, the label will be aligned with the slope of the line
       at the location of the label. If False, they will be horizontal.
    xvals : (xfirst, xlast) or array of float, optional
       The location of the labels. If a tuple, the labels will be
       evenly spaced between xfirst and xlast (in the axis units).
    drop_label : bool, optional
       If True, the label is consumed by the function so that subsequent
       calls to e.g. legend do not use it anymore.
    shrink_factor : double, optional
       Relative distance from the edges to place closest labels. Defaults to 0.05.
    yoffsets : number or list, optional.
        Distance relative to the line when positioning the labels. If given a number,
        the same value is used for all lines.
    outline_color : None | "auto" | color
        Colour of the outline. If set to "auto", use the background color.
        If set to None, do not draw an outline.
    outline_width : number
        Width of the outline
    kwargs : dict, optional
       Optional arguments passed to ax.text
    """
    ax = lines[0].axes

    handles, allLabels = ax.get_legend_handles_labels()

    all_lines = []
    for h in handles:
        if isinstance(h, ErrorbarContainer):
            all_lines.append(h.lines[0])
        else:
            all_lines.append(h)

    # In case no x location was provided, we need to use some heuristics
    # to generate them.
    if xvals is None:
        xvals = ax.get_xlim()
        xvals_rng = xvals[1] - xvals[0]
        shrinkage = xvals_rng * shrink_factor
        xvals = (xvals[0] + shrinkage, xvals[1] - shrinkage)

    if isinstance(xvals, tuple) and len(xvals) == 2:
        xmin, xmax = xvals
        xscale = ax.get_xscale()
        if xscale == "log":
            xvals = np.logspace(np.log10(xmin), np.log10(xmax), len(all_lines) + 2)[
                    1:-1
                    ]
        else:
            xvals = np.linspace(xmin, xmax, len(all_lines) + 2)[1:-1]

        # Build matrix line -> xvalue
        ok_matrix = np.zeros((len(all_lines), len(all_lines)), dtype=bool)

        for i, line in enumerate(all_lines):
            xdata = ensure_float(line.get_xdata())
            minx, maxx = min(xdata), max(xdata)
            for j, xv in enumerate(xvals):
                ok_matrix[i, j] = minx < xv < maxx

        # If some xvals do not fall in their corresponding line,
        # find a better matching using maximum bipartite matching.
        if not np.all(np.diag(ok_matrix)):
            order = maximum_bipartite_matching(ok_matrix)

            # The maximum match may miss a few points, let's add them back
            imax = order.max()
            order[order < 0] = np.arange(imax + 1, len(order))

            # Now reorder the xvalues
            old_xvals = xvals.copy()
            xvals[order] = old_xvals
    else:
        xvals = list(always_iterable(xvals))  # force the creation of a copy

    labLines, labels = [], []
    # Take only the lines which have labels other than the default ones
    for i, (line, xv) in enumerate(zip(all_lines, xvals)):
        label = allLabels[all_lines.index(line)]
        labLines.append(line)
        labels.append(label)

        # Move xlabel if it is outside valid range
        xdata = ensure_float(line.get_xdata())
        if not (min(xdata) <= xv <= max(xdata)):
            warnings.warn(
                (
                    "The value at position %s in `xvals` is outside the range of its "
                    "associated line (xmin=%s, xmax=%s, xval=%s). Clipping it "
                    "into the allowed range."
                )
                % (i, min(xdata), max(xdata), xv),
                UserWarning,
            )
            new_xv = min(xdata) + (max(xdata) - min(xdata)) * 0.9
            xvals[i] = new_xv

    # Convert float values back to datetime in case of datetime axis
    if isinstance(ax.xaxis.converter, DateConverter):
        xvals = [num2date(x).replace(tzinfo=ax.xaxis.get_units()) for x in xvals]

    txts = []
    try:
        yoffsets = [float(yoffsets)] * len(all_lines)
    except TypeError:
        pass
    for line, x, yoffset, label in zip(labLines, xvals, yoffsets, labels):
        txts.append(
            labelLine(
                line,
                x,
                label=label,
                align=align,
                drop_label=drop_label,
                yoffset=yoffset,
                outline_color=outline_color,
                outline_width=outline_width,
                **kwargs,
            )
        )

    return txts


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    x_plt = np.linspace(0, 2, 1000)
    y_plt = 0.1 * np.sin(2 * np.pi * x_plt)
    line, = plt.plot(x_plt, y_plt, label="hi")

    # plt.axis("equal")
    # print(ax.get_aspect())

    labelLine(line, x=1)
    plt.show()

    x = 1
    label = None
    align = True
    drop_label = False
    yoffset = 0
    yoffset_logspace = False
    outline_color = "auto"
    outline_width = 8