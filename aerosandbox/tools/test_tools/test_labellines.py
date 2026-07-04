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
