import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def plot_color_by_value(
    x: np.ndarray,
    y: np.ndarray,
    *args,
    c: np.ndarray,
    cmap="turbo",
    colorbar: bool = False,
    colorbar_label: str | None = None,
    clim: tuple[float, float] | None = None,
    **kwargs,
):
    """
    Plot a line that is multicolored based on an array of values.

    Uses the same syntax as matplotlib.pyplot.plot, except that `c` is now an array-like that
    maps to a specific color pulled from `cmap`. Makes lines that are multicolored based on
    this `c` value.

    Parameters
    ----------
    x : np.ndarray
        Array of x-points.
    y : np.ndarray
        Array of y-points.
    *args
        Args that will be passed into matplotlib.pyplot.plot().
        Example: ".-" for a dotted line.
    c : np.ndarray
        Array of values that will map to colors. Must be the same length as x and y.
    cmap
        The colormap to use.
    colorbar : bool
        Whether or not to display the colormap. [bool]
    colorbar_label : str | None
        The label to add to the colorbar. Only applies if the colorbar is created. [str]
    clim : tuple[float, float] | None
        A tuple of (min, max) that assigns bounds to the colormap. Computed from the range of
        `c` if not given.
    **kwargs
        Kwargs that will be passed into matplotlib.pyplot.plot().

    Returns
    -------
    tuple
        A tuple of (lines, sm, cbar), where:

        - lines: a list of the matplotlib Line2D objects that were plotted.
        - sm: the matplotlib ScalarMappable object corresponding to the colormap.
        - cbar: the colorbar object, if one was created (otherwise None).
    """
    cmap = mpl.colormaps.get_cmap(cmap)

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
        x1,
        x2,
        y1,
        y2,
        c1,
        c2,
    ) in enumerate(
        zip(
            x[:-1],
            x[1:],
            y[:-1],
            y[1:],
            c[:-1],
            c[1:],
        )
    ):
        line = plt.plot(
            [x1, x2],
            [y1, y2],
            *args,
            color=cmap(norm((c1 + c2) / 2) if cmin != cmax else 0.5),
            **kwargs,
        )
        lines += line

    if label is not None:
        line = plt.plot([None], [None], *args, color=cmap(0.5), label=label, **kwargs)
        lines += line
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    if colorbar:
        if colorbar_label is None:
            cbar = plt.colorbar(sm, ax=plt.gca())
        else:
            cbar = plt.colorbar(sm, ax=plt.gca(), label=colorbar_label)
    else:
        cbar = None
    return lines, sm, cbar


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    fig, ax = plt.subplots()
    x = np.linspace(-1, 1, 500)
    y = np.sin(10 * x)
    c = np.sin((10 * x) ** 2)
    plot_color_by_value(
        x, y, c=c, clim=(-1, 1), colorbar=True, colorbar_label="Colorbar Label"
    )
    p.show_plot(
        "Title",
        "X Axis",
        "Y Axis",
    )
