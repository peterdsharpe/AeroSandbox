from aerosandbox.geometry.airfoil.airfoil_families import (
    get_NACA_coordinates,
    get_UIUC_coordinates,
    get_kulfan_coordinates,
    get_kulfan_parameters,
)
import aerosandbox.numpy as np
import pytest


def test_get_NACA_coordinates():
    coords = get_NACA_coordinates(name="naca4408", n_points_per_side=100)
    assert len(coords) == 199


def test_get_UIUC_coordinates():
    coords = get_UIUC_coordinates(
        name="dae11",
    )
    assert len(coords) != 0


def test_get_kulfan_coordinates_default_not_mutated():
    """
    Using the deprecated `enforce_continuous_LE_radius` kwarg used to write into the module-level
    default `lower_weights` array, corrupting all subsequent calls that rely on the default.
    """
    before = get_kulfan_coordinates()

    with pytest.warns(DeprecationWarning):
        get_kulfan_coordinates(
            upper_weights=0.35 * np.ones(8),
            enforce_continuous_LE_radius=True,
        )

    after = get_kulfan_coordinates()

    assert np.allclose(before, after)


def test_get_kulfan_coordinates_caller_array_not_mutated():
    lower_weights = -0.3 * np.ones(8)
    lower_weights_original = lower_weights.copy()

    with pytest.warns(DeprecationWarning):
        get_kulfan_coordinates(
            lower_weights=lower_weights,
            upper_weights=0.35 * np.ones(8),
            enforce_continuous_LE_radius=True,
        )

    assert np.allclose(lower_weights, lower_weights_original)


def test_get_kulfan_parameters_opti_method():
    """
    get_kulfan_parameters(method="opti") used to crash with NameError (unimported `asb`).
    Check that it runs, and that the fitted parameters reconstruct the original airfoil closely.
    """
    from aerosandbox.geometry.airfoil import Airfoil

    af = Airfoil("naca2412").normalize()

    parameters = get_kulfan_parameters(
        coordinates=af.coordinates,
        method="opti",
    )

    af_reconstructed = Airfoil(
        name="Reconstructed",
        coordinates=get_kulfan_coordinates(**parameters),
    )

    x = np.linspace(0.01, 0.99, 21)
    assert (
        np.max(np.abs(af_reconstructed.local_thickness(x) - af.local_thickness(x)))
        < 5e-3
    )
    assert (
        np.max(np.abs(af_reconstructed.local_camber(x) - af.local_camber(x))) < 5e-3
    )


if __name__ == "__main__":
    pytest.main()
