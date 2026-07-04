"""
Tests for base-class behavior defined in aerosandbox/common.py.

These are placed here since there is no dedicated test folder for
aerosandbox/common.py; MassProperties (in this folder) is the canonical
concrete AeroSandboxObject used elsewhere to exercise base-class behavior.
"""

import inspect

import aerosandbox as asb
import pytest


class MyExplicitAnalysis(asb.ExplicitAnalysis):
    def __init__(self):
        pass


def test_get_options_returns_a_copy_of_geometry_options():
    """
    get_options() should never return the geometry object's stored options
    dict by reference; mutating the returned dict must not corrupt the
    geometry object.
    """
    wing = asb.Wing(
        name="Test Wing",
        analysis_specific_options={MyExplicitAnalysis: dict(foo=1)},
    )
    analysis = MyExplicitAnalysis()

    options = analysis.get_options(wing)
    assert options == {"foo": 1}

    options["foo"] = 999

    assert wing.analysis_specific_options[MyExplicitAnalysis]["foo"] == 1


if __name__ == "__main__":
    pytest.main([__file__])
