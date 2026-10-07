import warnings

import aerosandbox as asb
import pytest


def test_kulfan_airfoil_preserves_warning_filters():
    """
    KulfanAirfoil.__init__ used to call warnings.resetwarnings() (via an internal context manager),
    which wiped every globally-registered warning filter process-wide - including the user's and pytest's.
    """
    with warnings.catch_warnings():  # Sandbox this test's own filter manipulation.
        warnings.resetwarnings()
        warnings.simplefilter("error", FutureWarning)
        filters_before = list(warnings.filters)

        asb.KulfanAirfoil("naca2412")

        assert list(warnings.filters) == filters_before


if __name__ == "__main__":
    pytest.main([__file__])
