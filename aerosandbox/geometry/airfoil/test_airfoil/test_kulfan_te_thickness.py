"""Regression tests for KulfanAirfoil.TE_thickness matching Airfoil.TE_thickness() (issue #192)."""

from typing import Any

import pytest

import aerosandbox as asb
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_parameters


def get_airfoil_outputs(airfoil_name: str, n_weights: int = 60) -> dict[str, Any]:
    """Fit Kulfan parameters to a named coordinate airfoil (issue #192 snippet)."""
    foil = asb.Airfoil(airfoil_name)
    return get_kulfan_parameters(
        coordinates=foil.coordinates, n_weights_per_side=n_weights
    )


def _cadquery_te_thickness_call_site(airfoil, minimum_airfoil_TE_thickness: float = 0.001):
    """The TE-thickness check used by Airplane.generate_cadquery_geometry."""
    if airfoil.TE_thickness() < minimum_airfoil_TE_thickness:
        airfoil = airfoil.set_TE_thickness(thickness=minimum_airfoil_TE_thickness)
    return airfoil


def test_issue_192_kulfan_te_thickness_is_callable_like_airfoil():
    """The issue #192 snippet: KulfanAirfoil.TE_thickness() must work like Airfoil.TE_thickness()."""
    parameters = get_airfoil_outputs("naca0012")

    coordinate_te = asb.Airfoil("naca0012").TE_thickness()
    assert callable(asb.Airfoil("naca0012").TE_thickness)
    assert coordinate_te >= 0

    kaf = asb.KulfanAirfoil(
        upper_weights=parameters["upper_weights"],
        lower_weights=parameters["lower_weights"],
        leading_edge_weight=parameters["leading_edge_weight"],
        TE_thickness=parameters["TE_thickness"],
    )

    # Attribute-style access remains usable (legacy Kulfan field).
    attribute_value = kaf.TE_thickness
    # Method-style access matches Airfoil (this used to raise TypeError: 'float' is not callable).
    called_value = kaf.TE_thickness()

    assert callable(kaf.TE_thickness)
    assert called_value == pytest.approx(parameters["TE_thickness"])
    assert attribute_value == pytest.approx(parameters["TE_thickness"])
    assert called_value == pytest.approx(attribute_value)


def test_kulfan_te_thickness_constructor_argument_is_preserved():
    """The constructor still accepts TE_thickness= and stores that Kulfan parameter."""
    te = 0.012
    kaf = asb.KulfanAirfoil(
        lower_weights=[-0.1, -0.1, -0.05],
        upper_weights=[0.2, 0.3, 0.2],
        leading_edge_weight=0.1,
        TE_thickness=te,
    )
    assert kaf.TE_thickness() == pytest.approx(te)
    assert kaf.TE_thickness == pytest.approx(te)
    assert kaf.kulfan_parameters["TE_thickness"] == pytest.approx(te)

    kaf.TE_thickness = 0.02
    assert kaf.TE_thickness() == pytest.approx(0.02)
    assert kaf.kulfan_parameters["TE_thickness"] == pytest.approx(0.02)


def test_kulfan_te_thickness_liskov_substitutable_for_airfoil():
    """Code written against Airfoil.TE_thickness() must accept a KulfanAirfoil subclass."""

    def read_te_thickness(airfoil: asb.Airfoil) -> float:
        return airfoil.TE_thickness()

    coordinate_te = read_te_thickness(asb.Airfoil("naca0012"))
    kulfan_te = read_te_thickness(asb.KulfanAirfoil("naca0012"))

    assert coordinate_te >= 0
    assert kulfan_te >= 0
    assert kulfan_te == pytest.approx(
        asb.KulfanAirfoil("naca0012").kulfan_parameters["TE_thickness"]
    )


def test_generate_cadquery_geometry_te_thickness_call_site_with_kulfan_airfoil():
    """Airplane.generate_cadquery_geometry calls TE_thickness() then maybe set_TE_thickness()."""
    parameters = get_airfoil_outputs("naca0012")
    sharp = asb.KulfanAirfoil(
        upper_weights=parameters["upper_weights"],
        lower_weights=parameters["lower_weights"],
        leading_edge_weight=parameters["leading_edge_weight"],
        TE_thickness=0.0,
    )
    already_thick = asb.KulfanAirfoil(
        upper_weights=parameters["upper_weights"],
        lower_weights=parameters["lower_weights"],
        leading_edge_weight=parameters["leading_edge_weight"],
        TE_thickness=0.005,
    )

    thickened = _cadquery_te_thickness_call_site(
        sharp, minimum_airfoil_TE_thickness=0.001
    )
    unchanged = _cadquery_te_thickness_call_site(
        already_thick, minimum_airfoil_TE_thickness=0.001
    )

    assert thickened.TE_thickness() == pytest.approx(0.001)
    assert isinstance(thickened, asb.KulfanAirfoil)
    assert unchanged.TE_thickness() == pytest.approx(0.005)


def test_kulfan_te_thickness_accepts_optimization_variables():
    """Constructor TE_thickness= must still accept CasADi/Opti variables used in airfoil design."""
    opti = asb.Opti()
    te = opti.variable(init_guess=0.01, lower_bound=0)
    base = asb.KulfanAirfoil("naca0012")
    kaf = asb.KulfanAirfoil(
        lower_weights=base.lower_weights,
        upper_weights=base.upper_weights,
        leading_edge_weight=base.leading_edge_weight,
        TE_thickness=te,
    )

    coords = kaf.upper_coordinates()
    assert coords.shape[1] == 2
    # Method access returns the raw optimization variable (not a wrapper).
    assert kaf.TE_thickness() is te
    # Attribute arithmetic used by Kulfan geometry still unwraps to the variable.
    assert kaf.TE_thickness / 2 is not None
