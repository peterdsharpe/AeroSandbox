import warnings

import numpy as onp
import pytest

import aerosandbox as asb
import aerosandbox.numpy as np


def make_dyn() -> asb.DynamicsPointMass1DHorizontal:
    return asb.DynamicsPointMass1DHorizontal(
        mass_props=asb.MassProperties(mass=1),
        x_e=np.arange(10) ** 2,
        u_e=2 * np.arange(10),
    )


def test_array_conversion_no_numpy2_deprecation_warning():
    """
    NumPy 2.x calls __array__(dtype, copy=...); implementations that don't
    accept `copy` emit a DeprecationWarning and will raise TypeError in a
    future NumPy release.
    """
    dyn = make_dyn()

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        arr = onp.array(dyn)

    assert arr.shape == ()
    assert arr.item() is dyn


def test_array_conversion_with_explicit_copy_kwarg():
    dyn = make_dyn()
    arr = dyn.__array__(dtype="O", copy=True)
    assert arr.item() is dyn


if __name__ == "__main__":
    pytest.main([__file__])
