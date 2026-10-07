import numpy as np
import pytest

from aerosandbox.library.propulsion_turbofan import mass_turbofan


def mass_turbofan_reference(
    m_dot_core_corrected,
    overall_pressure_ratio,
    bypass_ratio,
    diameter_fan,
):
    """
    Independent reimplementation of the turbofan weight model, transcribed directly from
    the source document:

        Drela, M., "Turbofan Weight Model from Historical Data", 11 Nov 2016.
        (Also summarized in the TASOPT documentation, Section "Engine System Weight".)

    Bare weight: Eq. (1) with the frozen-exponent fit constants of Table 2:
        b_m = 1.0, b_pi = 1.0, b_alpha = 1.2,
        W_0 = 1684.5 lb, W_pi = 17.7 lb, W_alpha = 1662.2 lb

    Nacelle weight: Eqs. (21)-(30), with unit weights
        W_inlet = A_inlet (2.5 + 0.0238 d_fan)   [Eq. 22]
        W_fan   = A_fan   1.9                    [Eq. 23]
        W_exit  = A_exit  (2.5 + 0.0363 d_fan)   [Eq. 24]
        W_core  = A_core  1.9                    [Eq. 25]
    where areas are in ft^2, d_fan is in inches, and weights are in lb.

    Accessory / pylon weights: Eqs. (31)-(33) with f_add = f_pylon = 0.10.
    """
    kg_to_lbm = 2.20462262
    m_to_ft = 1 / 0.3048

    ### Bare weight, Eq. (1) + Table 2
    m_dot_lbm_per_sec = m_dot_core_corrected * kg_to_lbm
    W_bare_lbm = (m_dot_lbm_per_sec / 100) ** 1.0 * (
        1684.5
        + 17.7 * (overall_pressure_ratio / 30) ** 1.0
        + 1662.2 * (bypass_ratio / 5) ** 1.2
    )

    ### Nacelle weight, Eqs. (21)-(30)
    d_fan_ft = diameter_fan * m_to_ft
    d_fan_in = d_fan_ft * 12
    d_LPC_ft = d_fan_ft * bypass_ratio**-0.5

    S_nace_sqft = 12 * np.pi * (d_fan_ft / 2) ** 2
    A_inlet_sqft = 0.4 * S_nace_sqft
    A_fan_sqft = 0.2 * S_nace_sqft
    A_exit_sqft = 0.4 * S_nace_sqft
    A_core_sqft = 12 * np.pi * (d_LPC_ft / 2) ** 2

    W_nace_lbm = (
        A_inlet_sqft * (2.5 + 0.0238 * d_fan_in)
        + A_fan_sqft * 1.9
        + A_exit_sqft * (2.5 + 0.0363 * d_fan_in)
        + A_core_sqft * 1.9
    )

    ### Accessory + pylon weights, Eqs. (31)-(33)
    W_bare = W_bare_lbm / kg_to_lbm
    W_nace = W_nace_lbm / kg_to_lbm
    W_add = 0.10 * W_bare
    W_pylon = 0.10 * (W_bare + W_add + W_nace)

    return W_bare + W_add + W_nace + W_pylon


def test_mass_turbofan_matches_drela_source_equations():
    """
    Regression test for the nacelle exit-cowl unit weight: Drela's Eq. (24) is
    `W_exit = A_exit * (2.5 + 0.0363 * d_fan)` [d_fan in inches]. The code previously used
    `2.5 * 0.0363 * d_fan`, overweighting the exit cowl for fan diameters above ~46 in.
    """
    test_points = [
        # (m_dot_core_corrected [kg/s], OPR [-], BPR [-], d_fan [m])
        (364 / (5.95 + 1), 31.2, 5.95, 1.73),  # CFM56-2 class
        (20.0, 25.0, 4.0, 1.0),  # Smaller-engine point; pins diameter scaling
    ]
    for m_dot, opr, bpr, d_fan in test_points:
        assert mass_turbofan(
            m_dot_core_corrected=m_dot,
            overall_pressure_ratio=opr,
            bypass_ratio=bpr,
            diameter_fan=d_fan,
        ) == pytest.approx(
            mass_turbofan_reference(
                m_dot_core_corrected=m_dot,
                overall_pressure_ratio=opr,
                bypass_ratio=bpr,
                diameter_fan=d_fan,
            ),
            rel=1e-12,
        )


def test_mass_turbofan_pinned_value_cfm56_2():
    """
    Pins the model output for the CFM56-2-class example from the module's __main__ block.
    (Actual CFM56-2: ~2200 kg bare, ~3400 kg installed; the model gives ~3034 kg.)
    """
    mass = mass_turbofan(
        m_dot_core_corrected=364 / (5.95 + 1),
        overall_pressure_ratio=31.2,
        bypass_ratio=5.95,
        diameter_fan=1.73,
    )
    assert mass == pytest.approx(3034.05742523422, rel=1e-9)


if __name__ == "__main__":
    pytest.main([__file__])
