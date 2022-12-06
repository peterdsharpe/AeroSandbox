import matplotlib
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Union

preset_view_angles = {
    # Given in the form:
    #   * key is the view name
    #   * value is a tuple of three floats: (elev, azim, roll)
    'XY' : (90, -90, 0),
    'XZ' : (0, -90, 0),
    'YZ' : (0, 0, 0),
    '-XY': (-90, 90, 0),
    '-XZ': (0, 90, 0),
    '-YZ': (0, 180, 0)
}


def figure3d(
        *args,
        orthographic: bool = True,
        box_aspect: Tuple[float] = None,
        adjust_colors: bool = True,
        ax_kwargs: Dict = None,
        **kwargs
):
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
            f"Input '{preset_view}' is not a valid preset. Valid presets are:\n" +
            "\n".join([f"  * '{k}'" for k in preset_view_angles.keys()])
        )
    ax.view_init(
        elev=elev,
        azim=azim,
        roll=roll
    )


if __name__ == '__main__':
    figure3d()
