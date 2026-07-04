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


def test_general_aviation_mass_main_landing_gear():
    """
    Raymer Eq. 15.50: W_main_landing_gear = 0.095 (N_l W_l)^0.768 (L_m / 12)^0.409

    where "L_m = extended length of main landing gear, in." - so (L_m / 12) is
    the gear length in feet. Regression test for the length being divided by
    12 twice (i.e., inches / 144).

    C172-class inputs: W = 2450 lb, L_m = 26 in. Note that the current code
    computes N_l = n_gear * 1.5 = 3 (with the default n_gear=2) and uses the
    design TOGW for W_l; the hand computation below uses those same values.
    """
    mass = raymer_ga.mass_main_landing_gear(
        main_gear_length=26 * u.inch,
        design_mass_TOGW=2450 * u.lbm,
    )

    ### Hand computation, in Raymer's units (lb, in):
    # W_main_landing_gear = 0.095 * (3 * 2450)^0.768 * (26 / 12)^0.409
    #                     = 121.44 lb = 55.09 kg
    assert mass == pytest.approx(121.443 * u.lbm, rel=1e-3)


def test_general_aviation_mass_nose_landing_gear():
    """
    Raymer Eq. 15.51: W_nose_landing_gear = 0.125 (N_l W_l)^0.566 (L_n / 12)^0.845

    where "L_n = extended nose gear length, in." - so (L_n / 12) is the gear
    length in feet. Regression test for the length being divided by 12 twice.

    C172-class inputs: W = 2450 lb, L_n = 20 in. Note that the current code
    computes N_l = n_gear * 1.5 = 1.5 (with the default n_gear=1) and uses the
    design TOGW for W_l; the hand computation below uses those same values.
    """
    mass = raymer_ga.mass_nose_landing_gear(
        nose_gear_length=20 * u.inch,
        design_mass_TOGW=2450 * u.lbm,
    )

    ### Hand computation, in Raymer's units (lb, in):
    # W_nose_landing_gear = 0.125 * (1.5 * 2450)^0.566 * (20 / 12)^0.845
    #                     = 20.059 lb = 9.099 kg
    assert mass == pytest.approx(20.059 * u.lbm, rel=1e-3)


def test_general_aviation_landing_gear_casadi():
    """
    Checks that the GA landing-gear mass models also evaluate correctly when
    given CasADi symbolic inputs (dual-backend check).
    """
    import casadi

    opti = asb.Opti()
    main_gear_length = opti.variable(init_guess=26 * u.inch)
    nose_gear_length = opti.variable(init_guess=20 * u.inch)
    opti.subject_to(main_gear_length == 26 * u.inch)
    opti.subject_to(nose_gear_length == 20 * u.inch)

    mass_main = raymer_ga.mass_main_landing_gear(
        main_gear_length=main_gear_length,
        design_mass_TOGW=2450 * u.lbm,
    )
    mass_nose = raymer_ga.mass_nose_landing_gear(
        nose_gear_length=nose_gear_length,
        design_mass_TOGW=2450 * u.lbm,
    )
    assert isinstance(mass_main, casadi.MX)

    sol = opti.solve(verbose=False)

    assert sol(mass_main) == pytest.approx(121.443 * u.lbm, rel=1e-3)
    assert sol(mass_nose) == pytest.approx(20.059 * u.lbm, rel=1e-3)


if __name__ == "__main__":
    pytest.main([__file__])
