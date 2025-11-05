import aerosandbox.numpy as np
from aerosandbox.geometry import Wing, WingXSec, Airfoil
import pytest


def test_wing_simple_rectangular():
    """Test simple rectangular wing."""
    wing = Wing(
        xsecs=[
            WingXSec(xyz_le=[0, 0, 0], chord=1.0),
            WingXSec(xyz_le=[0, 5, 0], chord=1.0),
        ]
    )

    area = wing.area()
    span = wing.span()

    ### Rectangular wing: area = chord * span
    assert np.isclose(area, 1.0 * 5.0)
    assert np.isclose(span, 5.0)


def test_wing_tapered():
    """Test tapered wing."""
    wing = Wing(
        xsecs=[
            WingXSec(xyz_le=[0, 0, 0], chord=2.0),
            WingXSec(xyz_le=[0, 4, 0], chord=1.0),
        ]
    )

    area = wing.area()
    span = wing.span()

    ### Trapezoidal wing: area = (c1 + c2)/2 * span = (2 + 1)/2 * 4 = 6
    assert np.isclose(area, 6.0)
    assert np.isclose(span, 4.0)


def test_wing_symmetric():
    """Test symmetric wing doubles the area."""
    wing_half = Wing(
        symmetric=False,
        xsecs=[
            WingXSec(xyz_le=[0, 0, 0], chord=1.0),
            WingXSec(xyz_le=[0, 5, 0], chord=1.0),
        ],
    )

    wing_full = Wing(
        symmetric=True,
        xsecs=[
            WingXSec(xyz_le=[0, 0, 0], chord=1.0),
            WingXSec(xyz_le=[0, 5, 0], chord=1.0),
        ],
    )

    ### Symmetric wing should have twice the area
    assert np.isclose(wing_full.area(), 2 * wing_half.area())
    ### And twice the span
    assert np.isclose(wing_full.span(), 2 * wing_half.span())


def test_wing_sweep():
    """Test wing with sweep."""
    wing = Wing(
        xsecs=[
            WingXSec(xyz_le=[0, 0, 0], chord=1.0),
            WingXSec(xyz_le=[2, 5, 0], chord=1.0),  ### Swept back
        ]
    )

    ### Area should still be chord * span (projected)
    area = wing.area()
    assert area > 0


def test_wing_dihedral():
    """Test wing with dihedral."""
    wing = Wing(
        xsecs=[
            WingXSec(xyz_le=[0, 0, 0], chord=1.0),
            WingXSec(xyz_le=[0, 5, 1], chord=1.0),  ### Dihedral in z
        ]
    )

    ### Should still compute area
    area = wing.area()
    assert area > 0


def test_wing_twist():
    """Test wing with twist."""
    wing = Wing(
        xsecs=[
            WingXSec(xyz_le=[0, 0, 0], chord=1.0, twist=0),
            WingXSec(xyz_le=[0, 5, 0], chord=1.0, twist=5),  ### 5 degrees twist
        ]
    )

    ### Should handle twist
    area = wing.area()
    assert area > 0


def test_wing_mean_aerodynamic_chord():
    """Test mean aerodynamic chord calculation."""
    wing = Wing(
        xsecs=[
            WingXSec(xyz_le=[0, 0, 0], chord=2.0),
            WingXSec(xyz_le=[0, 4, 0], chord=1.0),
        ]
    )

    mac = wing.mean_aerodynamic_chord()

    ### For trapezoidal wing: MAC = (2/3) * (c_root + c_tip - c_root*c_tip/(c_root+c_tip))
    ### Simplified for this case
    assert 1.0 < mac < 2.0


def test_wing_aspect_ratio():
    """Test aspect ratio calculation."""
    wing = Wing(
        xsecs=[
            WingXSec(xyz_le=[0, 0, 0], chord=1.0),
            WingXSec(xyz_le=[0, 4, 0], chord=1.0),
        ]
    )

    ar = wing.aspect_ratio()
    span = wing.span()
    area = wing.area()

    ### AR = span^2 / area
    expected_ar = span**2 / area
    assert np.isclose(ar, expected_ar)


def test_wing_with_airfoil():
    """Test wing with specified airfoil."""
    airfoil = Airfoil("naca0012")

    wing = Wing(
        xsecs=[
            WingXSec(xyz_le=[0, 0, 0], chord=1.0, airfoil=airfoil),
            WingXSec(xyz_le=[0, 5, 0], chord=1.0, airfoil=airfoil),
        ]
    )

    ### Should create wing successfully
    assert wing.xsecs[0].airfoil.name == "naca0012"


def test_wing_multiple_sections():
    """Test wing with multiple sections."""
    wing = Wing(
        xsecs=[
            WingXSec(xyz_le=[0, 0, 0], chord=2.0),
            WingXSec(xyz_le=[0, 2, 0], chord=1.5),
            WingXSec(xyz_le=[0.5, 4, 0], chord=1.0),
            WingXSec(xyz_le=[1, 5, 0], chord=0.5),
        ]
    )

    area = wing.area()
    span = wing.span()

    ### Should handle multiple sections
    assert area > 0
    assert np.isclose(span, 5.0)


def test_wing_control_surface():
    """Test wing with control surface."""
    from aerosandbox.geometry.wing import ControlSurface

    cs = ControlSurface(name="flap", symmetric=True, hinge_point=0.75)

    wing = Wing(
        xsecs=[
            WingXSec(
                xyz_le=[0, 0, 0],
                chord=1.0,
                control_surfaces=[cs],
            ),
            WingXSec(
                xyz_le=[0, 5, 0],
                chord=1.0,
                control_surfaces=[cs],
            ),
        ]
    )

    ### Should create wing with control surface
    assert len(wing.xsecs[0].control_surfaces) == 1
    assert wing.xsecs[0].control_surfaces[0].symmetric == True


def test_wing_empty_xsecs():
    """Test wing with no xsecs."""
    wing = Wing(xsecs=[])

    ### Wing with no xsecs should have zero area and span
    area = wing.area()
    span = wing.span()
    assert np.isclose(area, 0.0)
    assert np.isclose(span, 0.0)


def test_wing_zero_chord():
    """Test wing with zero chord at tip."""
    wing = Wing(
        xsecs=[
            WingXSec(xyz_le=[0, 0, 0], chord=2.0),
            WingXSec(xyz_le=[0, 5, 0], chord=0.0),  ### Pointed tip
        ]
    )

    area = wing.area()

    ### Triangular wing: area = 0.5 * chord * span
    expected = 0.5 * 2.0 * 5.0
    assert np.isclose(area, expected)


def test_wing_variable_twist():
    """Test wing with varying twist distribution."""
    wing = Wing(
        xsecs=[
            WingXSec(xyz_le=[0, 0, 0], chord=1.0, twist=-2),
            WingXSec(xyz_le=[0, 2, 0], chord=1.0, twist=0),
            WingXSec(xyz_le=[0, 4, 0], chord=1.0, twist=2),
            WingXSec(xyz_le=[0, 5, 0], chord=1.0, twist=3),
        ]
    )

    ### Should handle variable twist
    area = wing.area()
    assert area > 0


def test_wing_c_shape():
    """Test wing with complex planform (C-shape curve)."""
    wing = Wing(
        xsecs=[
            WingXSec(xyz_le=[0, 0, 0], chord=1.0),
            WingXSec(xyz_le=[1, 2, 0.5], chord=0.8),
            WingXSec(xyz_le=[2, 3, 1.0], chord=0.6),
            WingXSec(xyz_le=[3, 3.5, 0.5], chord=0.4),
        ]
    )

    area = wing.area()
    ### Should compute area for complex shape
    assert area > 0


def test_wing_name():
    """Test wing with custom name."""
    wing = Wing(
        name="Main Wing",
        xsecs=[
            WingXSec(xyz_le=[0, 0, 0], chord=1.0),
            WingXSec(xyz_le=[0, 5, 0], chord=1.0),
        ],
    )

    assert wing.name == "Main Wing"


def test_wing_negative_span():
    """Test wing defined in negative y direction."""
    wing = Wing(
        symmetric=False,
        xsecs=[
            WingXSec(xyz_le=[0, 0, 0], chord=1.0),
            WingXSec(xyz_le=[0, -5, 0], chord=1.0),
        ],
    )

    ### Span should be absolute value
    span = wing.span()
    assert np.isclose(span, 5.0)


def test_wing_aspect_ratio_extreme():
    """Test aspect ratio for extreme cases."""
    ### High aspect ratio (glider-like)
    wing_high_ar = Wing(
        xsecs=[
            WingXSec(xyz_le=[0, 0, 0], chord=0.5),
            WingXSec(xyz_le=[0, 10, 0], chord=0.5),
        ]
    )

    ar_high = wing_high_ar.aspect_ratio()
    assert ar_high > 10

    ### Low aspect ratio (delta wing-like)
    wing_low_ar = Wing(
        xsecs=[
            WingXSec(xyz_le=[0, 0, 0], chord=5.0),
            WingXSec(xyz_le=[0, 2, 0], chord=0.0),
        ]
    )

    ar_low = wing_low_ar.aspect_ratio()
    assert ar_low < 2


def test_wing_taper_ratio():
    """Test taper ratio calculation."""
    wing = Wing(
        xsecs=[
            WingXSec(xyz_le=[0, 0, 0], chord=2.0),
            WingXSec(xyz_le=[0, 4, 0], chord=1.0),
        ]
    )

    taper_ratio = wing.taper_ratio()

    ### Taper ratio = tip_chord / root_chord
    expected = 1.0 / 2.0
    assert np.isclose(taper_ratio, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
