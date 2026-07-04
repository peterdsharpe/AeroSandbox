import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest


def test_star_import():
    """`__all__` should contain strings, so that star-imports work."""
    namespace = {}
    exec(
        "from aerosandbox.tools.pretty_plots.labellines import *",
        namespace,
    )
    assert callable(namespace["labelLine"])
    assert callable(namespace["labelLines"])


def test_labellines_labels_only_given_lines():
    """labelLines(lines=...) should label only the lines passed in."""
    from aerosandbox.tools.pretty_plots.labellines import labelLines

    fig, ax = plt.subplots()
    x = np.linspace(0, 1, 100)
    (l1,) = ax.plot(x, x, label="line1")
    (l2,) = ax.plot(x, x**2, label="line2")

    txts = labelLines(lines=[l1])
    assert [t.get_text() for t in txts] == ["line1"]

    plt.close(fig)


def test_labellines_all_lines():
    """Passing all lines labels all of them."""
    from aerosandbox.tools.pretty_plots.labellines import labelLines

    fig, ax = plt.subplots()
    x = np.linspace(0, 1, 100)
    ax.plot(x, x, label="line1")
    ax.plot(x, x**2, label="line2")

    txts = labelLines(lines=ax.get_lines())
    assert sorted(t.get_text() for t in txts) == ["line1", "line2"]

    plt.close(fig)


def test_labellines_empty_lines_list():
    """An empty list of lines should be a no-op, not an IndexError."""
    from aerosandbox.tools.pretty_plots.labellines import labelLines

    fig, ax = plt.subplots()
    txts = labelLines(lines=ax.get_lines())
    assert txts == []
    plt.close(fig)


def test_labellines_datetime_axis_no_deprecated_converter_access():
    """
    labelLines should work on a datetime x-axis without touching the
    deprecated `Axis.converter` attribute (deprecated in Matplotlib 3.10,
    removed in 3.12).
    """
    import datetime
    import warnings

    from aerosandbox.tools.pretty_plots.labellines import labelLines

    fig, ax = plt.subplots()
    dates = [
        datetime.datetime(2020, 1, 1) + datetime.timedelta(days=i) for i in range(10)
    ]
    ax.plot(dates, np.arange(10), label="my line")

    with warnings.catch_warnings():
        try:
            from matplotlib import MatplotlibDeprecationWarning

            warnings.simplefilter("error", MatplotlibDeprecationWarning)
        except ImportError:
            pass
        txts = labelLines(lines=ax.get_lines())

    assert [t.get_text() for t in txts] == ["my line"]
    plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__])
