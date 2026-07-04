import aerosandbox as asb
import aerosandbox.numpy as np
import pytest


def make_fuselage() -> asb.Fuselage:
    return asb.Fuselage(
        xsecs=[
            asb.FuselageXSec(xyz_c=[0, 0, 0], radius=0),
            asb.FuselageXSec(xyz_c=[0.5, 0, 0], radius=0.5),
            asb.FuselageXSec(xyz_c=[1, 0, 0], radius=0.3),
        ]
    )


def test_init():
    fuselage = make_fuselage()
    assert len(fuselage.xsecs) == 3
    assert fuselage.length() == pytest.approx(1.0)


def test_deprecated_symmetric_kwarg_raises():
    """
    The deprecated `symmetric` argument (captured via **kwargs) should raise,
    not be silently ignored.
    """
    with pytest.raises(DeprecationWarning):
        asb.Fuselage(symmetric=True)


def test_pending_deprecation_xyz_le_kwarg():
    """
    The pending-deprecation `xyz_le` argument (captured via **kwargs) should
    warn and translate the cross-sections, not be silently ignored.
    """
    with pytest.warns(match="xyz_le"):
        fuselage = asb.Fuselage(
            xsecs=[asb.FuselageXSec(xyz_c=[0, 0, 0], radius=1)],
            xyz_le=[1, 2, 3],
        )
    assert np.allclose(fuselage.xsecs[0].xyz_c, [1, 2, 3])


def test_unrecognized_kwarg_raises():
    with pytest.raises(TypeError):
        asb.Fuselage(not_a_real_kwarg=42)


if __name__ == "__main__":
    pytest.main()
