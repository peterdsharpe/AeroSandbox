import aerosandbox.numpy as np
import pytest


def test_softmax(plot=False):
    # Test softmax
    x = np.linspace(-10, 10, 100)
    y1 = x
    y2 = -2 * x - 3
    hardness = 0.5

    y_soft = np.softmax(y1, y2, hardness=hardness)

    assert np.softmax(0, 0, hardness=1) == np.log(2)

    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set(font_scale=1)

        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
        plt.plot(x, y1, label="y1")
        plt.plot(x, y2, label="y2")
        plt.plot(x, y_soft, label="softmax")
        plt.xlabel(r"x")
        plt.ylabel(r"y")
        plt.title(r"Softmax")
        plt.tight_layout()
        plt.legend()
        plt.show()


if __name__ == '__main__':
    pytest.main()
