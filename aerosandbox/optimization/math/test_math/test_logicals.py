import numpy as np
from aerosandbox.optimization import math
import pytest


def test_smoothmax(plot=False):
    # Test smoothmax
    x = math.linspace(-10, 10, 100)
    y1 = x
    y2 = -2 * x - 3
    hardness = 0.5

    ysmooth = math.smoothmax(y1, y2, hardness)

    assert math.smoothmax(0, 0, 1) == np.log(2)

    if plot:
        import matplotlib.pyplot as plt
        from matplotlib import style
        import seaborn as sns

        sns.set(font_scale=1)

        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
        plt.plot(x, y1, label="y1")
        plt.plot(x, y2, label="y2")
        plt.plot(x, ysmooth, label="smoothmax")
        plt.xlabel(r"x")
        plt.ylabel(r"y")
        plt.title(r"Smoothmax")
        plt.tight_layout()
        plt.legend()
        plt.show()


if __name__ == '__main__':
    pytest.main()
