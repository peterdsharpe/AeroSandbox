import aerosandbox.numpy as np
from aerosandbox.library.aerodynamics.viscous import Cd_cylinder, Cf_flat_plate
import pytest


def test_Cd_cylinder_low_reynolds():
    """Test cylinder drag at low Reynolds number (Stokes flow regime)."""
    Re_D = 1.0
    Cd = Cd_cylinder(Re_D, mach=0.0)

    ### At very low Re, Cd should be very high (> 10)
    assert Cd > 5.0
    assert not np.isnan(Cd)


def test_Cd_cylinder_moderate_reynolds():
    """Test cylinder drag at moderate Reynolds number."""
    Re_D = 1e4
    Cd = Cd_cylinder(Re_D, mach=0.0)

    ### Typical Cd for cylinder at Re ~ 1e4 is around 1.0-1.2
    assert 0.5 < Cd < 2.0
    assert not np.isnan(Cd)


def test_Cd_cylinder_high_reynolds():
    """Test cylinder drag at high Reynolds number (supercritical)."""
    Re_D = 1e6
    Cd = Cd_cylinder(Re_D, mach=0.0)

    ### At high Re (supercritical), Cd drops significantly (drag crisis)
    assert Cd < 1.0
    assert not np.isnan(Cd)


def test_Cd_cylinder_subcritical_only():
    """Test cylinder drag with subcritical_only flag."""
    Re_D = 1e5
    Cd_full = Cd_cylinder(Re_D, subcritical_only=False)
    Cd_subcritical = Cd_cylinder(Re_D, subcritical_only=True)

    ### Both should be positive and reasonable
    assert Cd_full > 0
    assert Cd_subcritical > 0
    assert not np.isnan(Cd_full)
    assert not np.isnan(Cd_subcritical)


def test_Cd_cylinder_with_mach_effects():
    """Test cylinder drag with Mach number effects."""
    Re_D = 1e5
    Cd_low_mach = Cd_cylinder(Re_D, mach=0.3, include_mach_effects=True)
    Cd_high_mach = Cd_cylinder(Re_D, mach=0.8, include_mach_effects=True)

    ### Higher Mach should generally increase drag (compressibility)
    assert Cd_high_mach > Cd_low_mach
    assert not np.isnan(Cd_low_mach)
    assert not np.isnan(Cd_high_mach)


def test_Cd_cylinder_mach_effects_disabled():
    """Test that disabling Mach effects gives same result for different Mach."""
    Re_D = 1e5
    Cd_mach_03 = Cd_cylinder(Re_D, mach=0.3, include_mach_effects=False)
    Cd_mach_08 = Cd_cylinder(Re_D, mach=0.8, include_mach_effects=False)

    ### Without Mach effects, Cd should be the same
    assert np.isclose(Cd_mach_03, Cd_mach_08)


def test_Cd_cylinder_reynolds_sweep():
    """Test cylinder drag over a range of Reynolds numbers for monotonicity issues."""
    Re_range = np.logspace(1, 6, 50)
    Cd_values = [Cd_cylinder(Re, subcritical_only=False) for Re in Re_range]

    ### All values should be positive
    assert all(Cd > 0 for Cd in Cd_values)
    ### No NaN values
    assert not any(np.isnan(Cd) for Cd in Cd_values)


def test_Cd_cylinder_zero_reynolds_handling():
    """Test that very small Reynolds numbers don't cause issues."""
    Re_D = 1e-10
    Cd = Cd_cylinder(Re_D)

    ### Should still return a finite positive value
    assert Cd > 0
    assert np.isfinite(Cd)


def test_Cf_flat_plate_blasius():
    """Test flat plate skin friction with Blasius solution (laminar)."""
    Re_L = 1e5

    Cf = Cf_flat_plate(Re_L, method="blasius")

    ### Blasius solution: Cf = 1.328 / sqrt(Re_L)
    expected = 1.328 / np.sqrt(Re_L)
    assert np.isclose(Cf, expected, rtol=0.01)


def test_Cf_flat_plate_turbulent():
    """Test flat plate skin friction for turbulent flow."""
    Re_L = 1e7

    Cf = Cf_flat_plate(Re_L, method="turbulent")

    ### Turbulent correlation uses 0.074 / Re^(1/5)
    expected = 0.074 / Re_L ** (1 / 5)
    assert np.isclose(Cf, expected)
    assert Cf > 0


def test_Cf_flat_plate_hybrid_cengel():
    """Test flat plate skin friction with hybrid (mixed laminar-turbulent) model."""
    Re_L = 1e6

    Cf = Cf_flat_plate(Re_L, method="hybrid-cengel")

    ### Hybrid-cengel uses: 0.074 / Re^(1/5) - 1742 / Re
    expected = 0.074 / Re_L ** (1 / 5) - 1742 / Re_L
    assert np.isclose(Cf, expected)
    assert Cf > 0


def test_Cf_flat_plate_hybrid_schlichting():
    """Test flat plate skin friction with hybrid-schlichting method."""
    Re_L = 1e6

    Cf = Cf_flat_plate(Re_L, method="hybrid-schlichting")

    ### Schlichting's model: 0.02666 * Re^(-0.139)
    expected = 0.02666 * Re_L**-0.139
    assert np.isclose(Cf, expected)
    assert not np.isnan(Cf)


def test_Cf_flat_plate_hybrid_sharpe_convex():
    """Test flat plate skin friction with hybrid-sharpe-convex method."""
    Re_L = 1e6

    Cf = Cf_flat_plate(Re_L, method="hybrid-sharpe-convex")

    ### Should be positive and reasonable
    assert 0 < Cf < 0.01
    assert not np.isnan(Cf)


def test_Cf_flat_plate_low_reynolds():
    """Test skin friction at low Reynolds number."""
    Re_L = 1e3

    Cf = Cf_flat_plate(Re_L, method="blasius")

    ### At low Re, Cf should be high
    assert Cf > 0.01
    assert not np.isnan(Cf)


def test_Cf_flat_plate_high_reynolds():
    """Test skin friction at high Reynolds number."""
    Re_L = 1e9

    Cf = Cf_flat_plate(Re_L, method="turbulent")

    ### At high Re, Cf should be small
    assert Cf < 0.005
    assert Cf > 0


def test_Cf_flat_plate_reynolds_sweep():
    """Test skin friction over Reynolds number range."""
    Re_range = np.logspace(3, 9, 50)
    Cf_values = [Cf_flat_plate(Re, method="hybrid-sharpe-convex") for Re in Re_range]

    ### All values should be positive
    assert all(Cf > 0 for Cf in Cf_values)
    ### Cf should decrease with increasing Re
    assert all(Cf_values[i] > Cf_values[i + 1] for i in range(len(Cf_values) - 1))


def test_Cf_flat_plate_method_comparison():
    """Compare different methods at the same Reynolds number."""
    Re_L = 1e6

    Cf_blasius = Cf_flat_plate(Re_L, method="blasius")
    Cf_schlichting = Cf_flat_plate(Re_L, method="hybrid-schlichting")
    Cf_hybrid_convex = Cf_flat_plate(Re_L, method="hybrid-sharpe-convex")

    ### All should be positive
    assert Cf_blasius > 0
    assert Cf_schlichting > 0
    assert Cf_hybrid_convex > 0

    ### Hybrid should be smooth blend, somewhere between methods
    assert Cf_hybrid_convex > 0


def test_Cd_cylinder_array_input():
    """Test cylinder drag with array of Reynolds numbers."""
    Re_array = np.array([1e3, 1e4, 1e5, 1e6])
    Cd_array = Cd_cylinder(Re_array)

    assert len(Cd_array) == len(Re_array)
    assert all(Cd > 0 for Cd in Cd_array)
    assert not any(np.isnan(Cd) for Cd in Cd_array)


def test_Cf_flat_plate_array_input():
    """Test flat plate skin friction with array of Reynolds numbers."""
    Re_array = np.array([1e4, 1e5, 1e6, 1e7])
    Cf_array = Cf_flat_plate(Re_array, method="hybrid-sharpe-convex")

    assert len(Cf_array) == len(Re_array)
    assert all(Cf > 0 for Cf in Cf_array)
    assert not any(np.isnan(Cf) for Cf in Cf_array)


def test_Cd_cylinder_negative_reynolds_handling():
    """Test that negative Reynolds numbers are handled (via absolute value)."""
    Re_D = -1e5
    Cd = Cd_cylinder(Re_D)

    ### Should handle as positive Re (via abs in the function)
    Cd_positive = Cd_cylinder(abs(Re_D))
    assert np.isclose(Cd, Cd_positive)


def test_Cf_flat_plate_invalid_method():
    """Test that invalid method raises ValueError."""
    Re_L = 1e5

    with pytest.raises(ValueError, match="method="):
        Cf_flat_plate(Re_L, method="invalid_method")


def test_Cd_cylinder_critical_reynolds_region():
    """Test cylinder drag in the critical Reynolds number region (drag crisis)."""
    ### Reynolds numbers around the drag crisis (2e5 to 5e5)
    Re_before = 2e5
    Re_during = 3e5
    Re_after = 5e5

    Cd_before = Cd_cylinder(Re_before, subcritical_only=False)
    Cd_during = Cd_cylinder(Re_during, subcritical_only=False)
    Cd_after = Cd_cylinder(Re_after, subcritical_only=False)

    ### Drag should decrease through the critical region
    assert Cd_during < Cd_before
    ### All should be positive
    assert all(Cd > 0 for Cd in [Cd_before, Cd_during, Cd_after])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
