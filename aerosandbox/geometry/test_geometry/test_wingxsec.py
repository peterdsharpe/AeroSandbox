from aerosandbox import WingXSec, Airfoil
import aerosandbox.numpy as np
import pytest


def test_init():
    xsec = WingXSec(
        xyz_le=np.array([0, 0, 0]),
        chord=1.0,
        twist=0,
        airfoil=Airfoil("naca0012"),
    )
    assert xsec.chord == 1.0
    assert xsec.twist == 0
    assert xsec.airfoil.name.lower() == "naca0012"
    assert xsec.control_surfaces == []


def test_deprecated_control_surface_kwargs():
    """
    The deprecated control_surface_* kwargs (captured via **deprecated_kwargs)
    should warn and build a ControlSurface, not be silently ignored.
    """
    with pytest.warns(match="control_surfaces"):
        xsec = WingXSec(
            chord=2,
            airfoil=Airfoil("naca0012"),
            control_surface_is_symmetric=False,
            control_surface_hinge_point=0.6,
            control_surface_deflection=10,
        )
    assert len(xsec.control_surfaces) == 1
    surface = xsec.control_surfaces[0]
    assert surface.symmetric is False
    assert surface.hinge_point == 0.6
    assert surface.deflection == 10

    ### Defaults should be filled in when only some deprecated kwargs are given
    with pytest.warns(match="control_surfaces"):
        xsec = WingXSec(
            chord=2,
            airfoil=Airfoil("naca0012"),
            control_surface_deflection=10,
        )
    assert len(xsec.control_surfaces) == 1
    surface = xsec.control_surfaces[0]
    assert surface.symmetric is True
    assert surface.hinge_point == 0.75
    assert surface.deflection == 10


def test_deprecated_twist_angle_kwarg():
    with pytest.warns(match="twist"):
        xsec = WingXSec(
            chord=2,
            airfoil=Airfoil("naca0012"),
            twist_angle=5,
        )
    assert xsec.twist == 5


def test_unrecognized_kwarg_raises():
    with pytest.raises(TypeError):
        WingXSec(
            chord=2,
            airfoil=Airfoil("naca0012"),
            not_a_real_kwarg=42,
        )


if __name__ == "__main__":
    pytest.main()
