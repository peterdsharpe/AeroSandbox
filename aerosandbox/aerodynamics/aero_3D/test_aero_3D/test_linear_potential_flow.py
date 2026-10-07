import aerosandbox as asb
import pytest
from aerosandbox.aerodynamics.aero_3D.linear_potential_flow import LinearPotentialFlow


@pytest.fixture
def airplane() -> asb.Airplane:
    return asb.Airplane(
        wings=[
            asb.Wing(
                symmetric=True,
                xsecs=[
                    asb.WingXSec(
                        xyz_le=[0, 0, 0], chord=1.0, airfoil=asb.Airfoil("naca0012")
                    ),
                    asb.WingXSec(
                        xyz_le=[0.2, 5, 0], chord=0.6, airfoil=asb.Airfoil("naca0012")
                    ),
                ],
            )
        ],
        fuselages=[
            asb.Fuselage(
                xsecs=[
                    asb.FuselageXSec(xyz_c=[x, 0, 0], radius=0.1 * x * (4 - x))
                    for x in [0, 1, 2, 3, 4]
                ]
            )
        ],
    )


def test_construction_with_defaults(airplane):
    """
    LinearPotentialFlow should be constructible with all-default arguments.

    (Regression test: this used to raise `TypeError: unhashable type: 'Wing'`, since the constructor
    built dicts keyed by Wing/Fuselage objects, which are unhashable.)
    """
    with pytest.warns(UserWarning):  # Class warns that it is under active development
        lpf = LinearPotentialFlow(
            airplane=airplane,
            op_point=asb.OperatingPoint(velocity=10),
        )

    wing = airplane.wings[0]
    fuselage = airplane.fuselages[0]

    assert lpf.wing_model[wing] == "vortex_lattice_all_horseshoe"
    assert lpf.fuselage_model[fuselage] == "none"

    # Default options should have been populated per-component:
    assert lpf.wing_options[wing]["spanwise_resolution"] == 10
    assert lpf.fuselage_options[fuselage] == {}


def test_construction_with_str_options(airplane):
    """
    User-specified `{str: value}` options should be applied to all components, merged over defaults.
    """
    with pytest.warns(UserWarning):
        lpf = LinearPotentialFlow(
            airplane=airplane,
            op_point=asb.OperatingPoint(velocity=10),
            wing_options={"chordwise_resolution": 5},
        )

    wing = airplane.wings[0]
    assert lpf.wing_options[wing]["chordwise_resolution"] == 5  # User-specified
    assert lpf.wing_options[wing]["spanwise_resolution"] == 10  # Default

    # The discretization (the currently-implemented part of the analysis) should build:
    discretization = lpf.discretization
    assert (
        len(discretization) == 1
    )  # 1 wing with a non-"none" model; fuselage model is "none"
    assert lpf.N_elements > 0


def test_invalid_wing_model_raises(airplane):
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError, match="Invalid wing model"):
            LinearPotentialFlow(
                airplane=airplane,
                op_point=asb.OperatingPoint(velocity=10),
                wing_model="not_a_real_model",
            )


def test_invalid_options_keys_raise(airplane):
    """
    Option dicts with keys that are neither str nor Wing/Fuselage should raise a descriptive ValueError.

    (Regression test: this used to raise `TypeError: issubclass() arg 1 must be a class`, since the
    validation called issubclass() on instances.)
    """
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError, match="`wing_options` must be"):
            LinearPotentialFlow(
                airplane=airplane,
                op_point=asb.OperatingPoint(velocity=10),
                wing_options={123: {"spanwise_resolution": 5}},
            )


if __name__ == "__main__":
    pytest.main([__file__])
