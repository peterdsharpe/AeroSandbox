import matplotlib.pyplot as plt


def figure3d(*args, orthographic=True, **kwargs):
    """
    Creates a new 3D figure. Args and kwargs are passed into matplotlib.pyplot.figure().

    Returns: (fig, ax)

    """
    fig = plt.figure(*args, **kwargs)

    axes_args = dict(
        projection='3d'
    )
    if orthographic:
        axes_args["proj_type"] = 'ortho'

    ax = plt.axes(**axes_args)
    return fig, ax


if __name__ == '__main__':
    figure3d()
