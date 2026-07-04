"""
Tests for base-class and package-level behavior defined in
aerosandbox/common.py and aerosandbox/__init__.py.

These are placed here since there is no dedicated test folder for
aerosandbox/common.py; MassProperties (in this folder) is the canonical
concrete AeroSandboxObject used elsewhere to exercise base-class behavior.
"""

import inspect
import warnings

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


def test_implicit_analysis_initialize_preserves_signature_and_docstring():
    """
    The @ImplicitAnalysis.initialize decorator should preserve the wrapped
    __init__'s signature and docstring (for help(), autodoc, and IDEs).
    """

    class MyImplicitAnalysis(asb.ImplicitAnalysis):
        @asb.ImplicitAnalysis.initialize
        def __init__(self, alpha: float = 5.0):
            """MyImplicitAnalysis docstring."""
            self.alpha = alpha

    parameters = inspect.signature(MyImplicitAnalysis.__init__).parameters
    assert "alpha" in parameters
    assert MyImplicitAnalysis.__init__.__doc__ == "MyImplicitAnalysis docstring."

    ### The decorator's injected behavior should be unchanged:
    analysis = MyImplicitAnalysis(alpha=2.0)
    assert analysis.alpha == 2.0
    assert analysis.opti_provided is False

    opti = asb.Opti()
    analysis = MyImplicitAnalysis(alpha=3.0, opti=opti)
    assert analysis.alpha == 3.0
    assert analysis.opti_provided is True
    assert analysis.opti is opti


def test_substitute_solution_deprecation_warning_points_at_caller():
    """
    substitute_solution's DeprecationWarning should carry stacklevel=2, so
    that it is attributed to the calling code (this file), not to
    aerosandbox/common.py.
    """
    mp = asb.MassProperties(mass=1.0)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mp.substitute_solution(sol=None)

    deprecations = [wi for wi in w if issubclass(wi.category, DeprecationWarning)]
    assert len(deprecations) == 1
    assert deprecations[0].filename == __file__


def test_docs_opens_hosted_documentation(monkeypatch):
    """
    asb.docs() should open the hosted documentation site (the Homepage
    declared in pyproject.toml), not the GitHub source tree.
    """
    import webbrowser

    opened_urls = []
    monkeypatch.setattr(webbrowser, "open_new", opened_urls.append)

    asb.docs()

    assert opened_urls == ["https://peterdsharpe.github.io/AeroSandbox/"]


if __name__ == "__main__":
    pytest.main([__file__])
