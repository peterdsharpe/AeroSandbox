import matplotlib.pyplot as plt

def hline(
        y,
        linestyle="--",
        color="k",
        text: str = None,
        text_xloc=0.5,
        text_ha="center",
        text_va="bottom",
        text_kwargs=None,
):  # TODO docs
    if text_kwargs is None:
        text_kwargs = {}
    ax = plt.gca()
    xlim = ax.get_xlim()
    plt.axhline(y=y, ls=linestyle, color=color)
    if text is not None:
        plt.text(
            x=text_xloc * xlim[1] + (1 - text_xloc) * xlim[0],
            y=y,
            s=text,
            color=color,
            horizontalalignment=text_ha,
            verticalalignment=text_va,
            **text_kwargs
        )


def vline(
        x,
        linestyle="--",
        color="k",
        text: str = None,
        text_yloc=0.5,
        text_ha="right",
        text_va="center",
        text_kwargs=None,
):  # TODO docs
    if text_kwargs is None:
        text_kwargs = {}
    ax = plt.gca()
    ylim = ax.get_ylim()
    plt.axvline(x=x, ls=linestyle, color=color)
    if text is not None:
        plt.text(
            x=x,
            y=text_yloc * ylim[1] + (1 - text_yloc) * ylim[0],
            s=text,
            color=color,
            horizontalalignment=text_ha,
            verticalalignment=text_va,
            rotation=90,
            **text_kwargs
        )
