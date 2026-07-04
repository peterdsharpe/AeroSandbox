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


if __name__ == "__main__":
    pytest.main([__file__])
