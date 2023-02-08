import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

N = 10000

def get_xy(s):
    theta = np.linspace(0, np.pi / 2, N)
    st = np.sin(theta)
    ct = np.cos(theta)

    ct[-1] = 0
    st[0] = 0

    x = ct ** (2/s)
    y = st ** (2/s)

    return x, y

def plot_xy(s):
    x, y = get_xy(s)
    fig, ax = plt.subplots()
    plt.plot(x, y)
    p.equal()
    p.show_plot()

### Generate data
@np.vectorize
def get_arc_length(s):
    x, y = get_xy(s)

    dx = np.diff(x)
    dy = np.diff(y)

    darc = (dx ** 2 + dy ** 2) ** 0.5

    arc_length = np.sum(darc)

    return arc_length

s = np.concatenate([
    np.linspace(1e-6, 3, 200),
    np.linspace(3, 50, 100)[1:]
])
arc_lengths = get_arc_length(s)