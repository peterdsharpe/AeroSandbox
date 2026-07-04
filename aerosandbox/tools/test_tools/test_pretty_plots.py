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


def test_pie_sort_by_array():
    """
    `sort_by` is documented to accept "an array of numbers corresponding to
    each pie slice"; passing a NumPy array should not crash.
    """
    fig, ax = plt.subplots()
    p.pie(
        values=[3, 1, 2],
        names=["a", "b", "c"],
        sort_by=np.array([2, 0, 1]),
    )
    plt.close(fig)


def test_pie_sort_by_string_options():
    for sort_by in ["values", "names", None]:
        fig, ax = plt.subplots()
        p.pie(
            values=[3, 1, 2],
            names=["a", "b", "c"],
            sort_by=sort_by,
        )
        plt.close(fig)


def test_pie_sort_by_invalid_string():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        p.pie(
            values=[3, 1, 2],
            names=["a", "b", "c"],
            sort_by="not_a_valid_option",
        )
    plt.close(fig)


def test_contour_z_log_scale_constant_z():
    """contour(z_log_scale=True) should not crash when Z is constant."""
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    Z = 5.0 * np.ones_like(X)
    p.contour(X, Y, Z, z_log_scale=True)
    plt.close(fig)


def test_contour_z_log_scale_normal():
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    Z = 10 ** (X + Y)
    p.contour(X, Y, Z, z_log_scale=True)
    plt.close(fig)


def test_contour_z_log_scale_nonpositive_z_raises():
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    Z = X - 0.5  # Contains nonpositive values
    with pytest.raises(ValueError, match="positive"):
        p.contour(X, Y, Z, z_log_scale=True)
    plt.close(fig)


def test_get_last_line_color():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], color="darkorchid")
    (line,) = ax.plot([0, 1], [1, 0], color="crimson")
    assert p.get_last_line_color() == line.get_color() == "crimson"
    plt.close(fig)


def test_get_last_line_color_no_lines():
    """With no lines on the axes, falls back to a palette color without crashing."""
    fig, ax = plt.subplots()
    color = p.get_last_line_color()
    assert color is not None
    plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__])
