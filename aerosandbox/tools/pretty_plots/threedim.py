import matplotlib
import mpl_toolkits
import matplotlib.pyplot as plt
import aerosandbox.numpy as np
from typing import Dict, Tuple, Union

preset_view_angles = {
    # Given in the form:
    #   * key is the view name
    #   * value is a tuple of three floats: (elev, azim, roll)
    'XY'             : (90, -90, 0),
    'XZ'             : (0, -90, 0),
    'YZ'             : (0, 0, 0),
    '-XY'            : (-90, 90, 0),
    '-XZ'            : (0, 90, 0),
    '-YZ'            : (0, 180, 0),
    'left_isometric' : (np.arctan2d(1, 2 ** 0.5), -135, 0),
    'right_isometric': (np.arctan2d(1, 2 ** 0.5), 135, 0)
}


def figure3d(
        *args,
        orthographic: bool = True,
        box_aspect: Tuple[float] = None,
        adjust_colors: bool = True,
        ax_kwargs: Dict = None,
        **kwargs
) -> Tuple[matplotlib.figure.Figure, mpl_toolkits.mplot3d.axes3d.Axes3D]:
    """
    Creates a new 3D figure. Args and kwargs are passed into matplotlib.pyplot.figure().

    Returns: (fig, ax)

    """
    ### Set defaults
    if ax_kwargs is None:
        ax_kwargs = {}

    ### Generate the figure
    fig = plt.figure(*args, **kwargs)

    ### Generate the axes
    default_axes_kwargs = dict(
        projection='3d',
        proj_type='ortho' if orthographic else 'persp',
        box_aspect=box_aspect,
    )
    axes_kwargs = {  # Overwrite any of the computed kwargs with user-provided ones, where applicable.
        **default_axes_kwargs,
        **ax_kwargs,
    }

    ax = fig.add_subplot(**axes_kwargs)

    if adjust_colors:
        pane_color = ax.get_facecolor()
        ax.set_facecolor((0, 0, 0, 0))  # Set transparent

        ax.xaxis.pane.set_facecolor(pane_color)
        ax.xaxis.pane.set_alpha(1)
        ax.yaxis.pane.set_facecolor(pane_color)
        ax.yaxis.pane.set_alpha(1)
        ax.zaxis.pane.set_facecolor(pane_color)
        ax.zaxis.pane.set_alpha(1)

    return fig, ax


def ax_is_3d(
        ax: matplotlib.axes.Axes = None
) -> bool:
    """
    Determines if a Matplotlib axis object is 3D or not.

    Args:
        ax: The axis object. If not given, uses the current active axes.

    Returns: A boolean of whether the axis is 3D or not.

    """
    if ax is None:
        ax = plt.gca()

    return hasattr(ax, 'zaxis')


def set_preset_3d_view_angle(
        preset_view: str
) -> None:
    ax = plt.gca()

    if not ax_is_3d(ax):
        raise Exception("Can't set a 3D view angle on a non-3D plot!")

    try:
        elev, azim, roll = preset_view_angles[preset_view]
    except KeyError:
        raise ValueError(
            f"Input '{preset_view}' is not a valid preset view. Valid presets are:\n" +
            "\n".join([f"  * '{k}'" for k in preset_view_angles.keys()])
        )

    if roll == 0:
        # This is to maintain back-compatibility to older Matplotlib versions.
        # Older versions of Matplotlib (roughly, <=3.4.0) didn't support the `roll` kwarg.
        # Hence, if we don't need to edit the roll, we don't - this extends back-compatibility.
        ax.view_init(
            elev=elev,
            azim=azim,
        )
    else:
        ax.view_init(
            elev=elev,
            azim=azim,
            roll=roll
        )


if __name__ == '__main__':
    import aerosandbox.numpy as np
    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    figure3d()

    t = np.linspace(0, 1, 100)

    x = np.sin(4 * 2 * np.pi * t)
    y = t ** 2
    z = 5 * t

    fig, ax = p.figure3d()
    p.set_preset_3d_view_angle('-YZ')

    ax.plot(
        x, y, z, "-"
    )
    p.set_ticks(*([1, 1] * 3))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    p.equal()
    p.show_plot()
