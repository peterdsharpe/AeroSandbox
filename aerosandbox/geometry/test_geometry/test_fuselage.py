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


def test_fineness_ratio():
    """
    Builds an exact Sears-Haack body of known fineness ratio and checks
    Fuselage.fineness_ratio() against it. The fineness ratio is defined as
    length / max_diameter (not length / max_radius).
    """
    length = 10
    r_max = 1.0
    true_fineness_ratio = length / (2 * r_max)

    x = np.sinspace(0, length, 501)
    x_nondim = x / length
    r = r_max * (4 * x_nondim * (1 - x_nondim)) ** (3 / 4)  # Sears-Haack shape
    fuselage = asb.Fuselage(
        xsecs=[
            asb.FuselageXSec(xyz_c=[xi, 0, 0], radius=ri) for xi, ri in zip(x, r)
        ]
    )

    assert fuselage.fineness_ratio(assumed_shape="sears-haack") == pytest.approx(
        true_fineness_ratio, rel=1e-3
    )
    ### Alternate spelling used by callers elsewhere in the codebase:
    assert fuselage.fineness_ratio(assumed_shape="sears_haack") == pytest.approx(
        true_fineness_ratio, rel=1e-3
    )
    ### The cylinder assumption should not depend on the radius distribution:
    cylinder = asb.Fuselage(
        xsecs=[
            asb.FuselageXSec(xyz_c=[0, 0, 0], radius=r_max),
            asb.FuselageXSec(xyz_c=[length, 0, 0], radius=r_max),
        ]
    )
    assert cylinder.fineness_ratio(assumed_shape="cylinder") == pytest.approx(
        true_fineness_ratio
    )

    with pytest.raises(ValueError):
        fuselage.fineness_ratio(assumed_shape="not-a-shape")


if __name__ == "__main__":
    pytest.main()
