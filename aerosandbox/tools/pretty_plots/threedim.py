import matplotlib.pyplot as plt


def figure3d(*args, **kwargs):
    """
    Creates a new 3D figure. Args and kwargs are passed into matplotlib.pyplot.figure().

    Returns: (fig, ax)

    """
    fig = plt.figure(*args, **kwargs)
    ax = plt.axes(projection='3d')
    return fig, ax
