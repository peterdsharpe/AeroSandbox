import matplotlib.colors as mc
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
}


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
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
