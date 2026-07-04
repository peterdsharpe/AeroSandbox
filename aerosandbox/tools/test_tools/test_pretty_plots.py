import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import aerosandbox.tools.pretty_plots as p


def test_show_plot_legend_inline_with_colorbar_axes():
    """
    show_plot(legend_inline=True) should not crash when the figure contains
    an axes with no lines (e.g., a colorbar axes).
    """
    fig, ax = plt.subplots()
    x = np.linspace(0, 1, 100)
    ax.plot(x, x, label="a")
    ax.plot(x, x**2, label="b")

    sm = plt.cm.ScalarMappable(cmap="viridis")
    plt.colorbar(sm, ax=ax)

    p.show_plot(show=False, legend_inline=True)

    plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__])
