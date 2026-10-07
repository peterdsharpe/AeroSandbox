import matplotlib.pyplot as plt
import matplotlib.transforms as transforms


def hline(
    y,
    linestyle="--",
    color="k",
    text: str | None = None,
    text_xloc=0.5,
    text_ha="center",
    text_va="bottom",
    text_kwargs=None,
    **kwargs,
):  # TODO docs
    """
    Draw a horizontal line on the current axes, with an optional text label.

    Parameters
    ----------
    y
        The y-value at which the line is drawn.
    linestyle
        The linestyle of the line.
    color
        The color of the line (and of the text, if any).
    text : str | None
        If given, a text label to draw on the line.
    text_xloc
        The x-location of the text, in axes (0 to 1) coordinates.
    text_ha
        The horizontal alignment of the text.
    text_va
        The vertical alignment of the text.
    text_kwargs
        Additional keyword arguments passed to `plt.annotate()`.
    **kwargs
        Additional keyword arguments passed to `plt.axhline()`.
    """
    if text_kwargs is None:
        text_kwargs = {}
    ax = plt.gca()
    plt.axhline(y=y, ls=linestyle, color=color, **kwargs)
    if text is not None:
        trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        plt.annotate(
            text=text,
            xy=(text_xloc, y),
            xytext=(0, 0),
            xycoords=trans,
            textcoords="offset points",
            ha=text_ha,
            va=text_va,
            color=color,
            **text_kwargs,
        )


def vline(
    x,
    linestyle="--",
    color="k",
    text: str | None = None,
    text_yloc=0.5,
    text_ha="right",
    text_va="center",
    text_kwargs=None,
    **kwargs,
):  # TODO docs
    """
    Draw a vertical line on the current axes, with an optional text label.

    Parameters
    ----------
    x
        The x-value at which the line is drawn.
    linestyle
        The linestyle of the line.
    color
        The color of the line (and of the text, if any).
    text : str | None
        If given, a text label to draw on the line.
    text_yloc
        The y-location of the text, in axes (0 to 1) coordinates.
    text_ha
        The horizontal alignment of the text.
    text_va
        The vertical alignment of the text.
    text_kwargs
        Additional keyword arguments passed to `plt.annotate()`.
    **kwargs
        Additional keyword arguments passed to `plt.axvline()`.
    """
    if text_kwargs is None:
        text_kwargs = {}
    ax = plt.gca()
    plt.axvline(x=x, ls=linestyle, color=color, **kwargs)
    if text is not None:
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        plt.annotate(
            text=text,
            xy=(x, text_yloc),
            xytext=(0, 0),
            xycoords=trans,
            textcoords="offset points",
            ha=text_ha,
            va=text_va,
            color=color,
            rotation=90,
            **text_kwargs,
        )
