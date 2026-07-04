"""
Regression tests for the Raymer weight-buildup models in
`aerosandbox/library/weights/raymer_cargo_transport_weights.py` and
`aerosandbox/library/weights/raymer_general_aviation_weights.py`.

(Placed in this directory as the nearest existing test directory to
`aerosandbox/library/weights/`, which has no test directory of its own.)

Expected values are computed by hand, directly from the equations in Raymer,
"Aircraft Design: A Conceptual Approach" (6th Ed., 2018), Sections 15.3.2
("Cargo/Transport Weights") and 15.3.3 ("General Aviation Weights"). Raymer's
equations use British units (lengths in ft unless otherwise noted; weights in
lb) and give weights in lb; the AeroSandbox functions take and return SI units.
"""

import pytest

import aerosandbox as asb
import aerosandbox.tools.units as u
from aerosandbox.library.weights import raymer_cargo_transport_weights as raymer_ct
from aerosandbox.library.weights import raymer_general_aviation_weights as raymer_ga


def make_transport_geometry():
    """
    Makes 737-800-class fuselage + main wing geometry:
    fuselage length 129.5 ft, wing span 112.6 ft, with two control-surface
    functions (aileron, flap) on the main wing.
    """
    fuselage = asb.Fuselage(
        xsecs=[
            asb.FuselageXSec(xyz_c=[0, 0, 0], radius=1.88),
            asb.FuselageXSec(xyz_c=[129.5 * u.foot, 0, 0], radius=1.88),
        ]
    )
    main_wing = asb.Wing(
        symmetric=True,
        xsecs=[
            asb.WingXSec(
                xyz_le=[0, 0, 0],
                chord=7.32,
                airfoil=asb.Airfoil("naca2412"),
                control_surfaces=[
                    asb.ControlSurface(name="Aileron"),
                    asb.ControlSurface(name="Flap"),
                ],
            ),
            asb.WingXSec(
                xyz_le=[4, 112.6 / 2 * u.foot, 0],
                chord=1.26,
                airfoil=asb.Airfoil("naca2412"),
            ),
        ],
    )
    return fuselage, main_wing


def test_cargo_transport_mass_nacelles_returns_kg():
    """
    Raymer Eq. 15.31 (Section 15.3.2, "British Units, Results in Pounds"):

        W_nacelle_group = 0.6724 K_ng N_Lt^0.10 N_w^0.294 N_z^0.119
                          * W_ec^0.611 N_en^0.984 S_n^0.224

    Checks that the result is correctly converted from lb to kg (regression
    test for a missing `* u.lbm` factor). 737/CFM56-class inputs.
    """
    mass = raymer_ct.mass_nacelles(
        nacelle_length=15 * u.foot,
        nacelle_width=7 * u.foot,
        nacelle_height=7 * u.foot,
        ultimate_load_factor=3.75,
        mass_per_engine=4301 * u.lbm,
        n_engines=2,
        is_pylon_mounted=True,
        engines_have_thrust_reversers=True,
    )

    ### Hand computation, in Raymer's units (lb, ft):
    # W_ec = 2.331 * W_en^0.901 * K_p * K_tr = 2.331 * 4301^0.901 * 1 * 1.18
    #      = 5167.4 lb
    # S_n (per the code's wetted-area model) = 2 * (15 * 7) + 2 * (7 * 7)
    #      = 308 ft^2
    # W_nacelle_group = 0.6724 * 1.017 * 15^0.10 * 7^0.294 * 3.75^0.119
    #                   * 5167.4^0.611 * 2^0.984 * 308^0.224
    #                 = 2464.8 lb = 1118.0 kg
    #
    # Loose relative tolerance since the code (intentionally) smooths W_ec
    # with a softmax against the bare engine mass; this tolerance is far
    # smaller than the 2.205x error from a missing lb -> kg conversion.
    assert mass == pytest.approx(2464.8 * u.lbm, rel=0.02)


def test_cargo_transport_mass_instruments():
    """
    Raymer Eq. 15.37: W_instruments = 4.509 K_r K_tp N_c^0.541 N_en (L_f + B_w)^0.5

    Regression test for (L_f + B_w) mistyped as (L_f * B_w).
    """
    fuselage, main_wing = make_transport_geometry()

    mass = raymer_ct.mass_instruments(
        fuselage=fuselage,
        main_wing=main_wing,
        n_engines=2,
        n_crew=2,
    )

    ### Hand computation, in Raymer's units (lb, ft):
    # W_instruments = 4.509 * 1 * 1 * 2^0.541 * 2 * (129.5 + 112.6)^0.5
    #               = 204.16 lb = 92.60 kg
    assert mass == pytest.approx(204.157 * u.lbm, rel=1e-3)


def test_cargo_transport_mass_hydraulics():
    """
    Raymer Eq. 15.38: W_hydraulics = 0.2673 N_f (L_f + B_w)^0.937

    Regression test for (L_f + B_w) mistyped as (L_f * B_w).
    """
    fuselage, main_wing = make_transport_geometry()
    airplane = asb.Airplane(wings=[main_wing], fuselages=[fuselage])

    mass = raymer_ct.mass_hydraulics(
        airplane=airplane,
        fuselage=fuselage,
        main_wing=main_wing,
    )

    ### Hand computation, in Raymer's units (lb, ft), with
    ### N_f = 2 control-surface functions (aileron, flap):
    # W_hydraulics = 0.2673 * 2 * (129.5 + 112.6)^0.937
    #              = 91.59 lb = 41.54 kg
    assert mass == pytest.approx(91.587 * u.lbm, rel=1e-3)


if __name__ == "__main__":
    pytest.main([__file__])
