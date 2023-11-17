import matplotlib.colors as mc
import matplotlib as mpl
import matplotlib.pyplot as plt
import aerosandbox.numpy as np
import colorsys

palettes = {
    "categorical": [
        "#4285F4",  # From Google logo, blue
        "#EA4335",  # From Google logo, red
        "#34A853",  # From Google logo, green
        "#ECB22E",  # From Slack logo, gold
        "#9467BD",  # From Matplotlib "tab10", purple
        "#8C564B",  # From Matplotlib "tab10", brown
        "#E377C2",  # From Matplotlib "tab10", pink
        "#7F7F7F",  # From Matplotlib "tab10", gray
    ],
    "pales": [
        "#C87A7A",
        "#B1866E",
        "#9B8E6E",
        "#90926E",
        "#85956D",
        "#77986D",
        "#6E9973",
        "#6F9882",
        "#719690",
        "#73959C",
        "#7692B1",
        "#868CBC",
    ]
}


def get_discrete_colors_from_colormap(
        cmap: str = "rainbow",
        N: int = 8,
        lower_bound: float = 0,
        upper_bound: float = 1,
):
    """
    Returns uniformly-spaced discrete color samples from a (continuous) colormap.
    Args:
        cmap: Name of a colormap.
        N: Number of colors to retrieve from colormap.

    Returns: A Nx4 array of (R, G, B, A) colors.

    """
    cmap = mpl.colormaps.get_cmap(cmap)
    colors = cmap(np.linspace(lower_bound, upper_bound, N))
    return colors


def adjust_lightness(color, amount=1.0):
    """
    Converts a color to HLS space, then multiplies the lightness by `amount`, then converts back to RGB.

    Args:
        color: A color, in any format that matplotlib understands.
        amount: The amount to multiply the lightness by. Valid range is 0 to infinity.

    Returns: A color, as an RGB tuple.

    """
    try:
        c = mc.cnames[color]
    except Exception:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def get_last_line_color():
    lines = plt.gca().lines
    try:
        line = lines[-1]
        return line._color
    except IndexError:
        return palettes["categorical"][0] # TODO make this just the first color in the current palette
