import matplotlib.pyplot as plt

def figure3d(*args, **kwargs):
    fig = plt.figure(*args, **kwargs)
    ax = plt.axes(projection='3d')
    return fig, ax
