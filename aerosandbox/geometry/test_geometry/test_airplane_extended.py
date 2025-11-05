import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.geometry import (
    Airplane,
    Wing,
    WingXSec,
    Fuselage,
    FuselageXSec,
    Propulsor,
)
import pytest


def test_airplane_initialization_empty():
    """Test that creating an airplane with no components raises error."""
    with pytest.raises(ValueError):
        Airplane()


def test_airplane_initialization_with_name():
    """Test creating an airplane with a custom name and explicit reference values."""
    airplane = Airplane(name="TestPlane", s_ref=10, c_ref=1, b_ref=10)

    assert airplane.name == "TestPlane"
    assert airplane.s_ref == 10


def test_airplane_with_single_wing():
    """Test airplane with one wing, checking reference values."""
    wing = Wing(
        xsecs=[
            WingXSec(xyz_le=[0, 0, 0], chord=1.0),
            WingXSec(xyz_le=[0, 1, 0], chord=0.5),
        ]
    )

    airplane = Airplane(wings=[wing])

    ### Should auto-compute s_ref from wing area
    assert airplane.s_ref > 0
    assert airplane.c_ref > 0
    assert airplane.b_ref > 0


def test_airplane_with_multiple_wings():
    """Test airplane with multiple wings."""
    wing1 = Wing(
        xsecs=[
            WingXSec(xyz_le=[0, 0, 0], chord=1.0),
            WingXSec(xyz_le=[0, 1, 0], chord=0.5),
        ]
    )
    wing2 = Wing(
        xsecs=[
            WingXSec(xyz_le=[3, 0, 0], chord=0.5),
            WingXSec(xyz_le=[3, 0.5, 0], chord=0.25),
        ]
    )

    airplane = Airplane(wings=[wing1, wing2])

    assert len(airplane.wings) == 2
    ### Reference values should come from first wing
    assert airplane.s_ref == wing1.area()


def test_airplane_with_fuselage():
    """Test airplane with fuselage."""
    fuselage = Fuselage(
        xsecs=[
            FuselageXSec(xyz_c=[0, 0, 0], radius=0.5),
            FuselageXSec(xyz_c=[5, 0, 0], radius=0.5),
        ]
    )

    airplane = Airplane(fuselages=[fuselage])

    assert len(airplane.fuselages) == 1


def test_airplane_with_wing_and_fuselage():
    """Test airplane with both wing and fuselage."""
    wing = Wing(
        xsecs=[
            WingXSec(xyz_le=[0, 0, 0], chord=1.0),
            WingXSec(xyz_le=[0, 1, 0], chord=0.5),
        ]
    )
    fuselage = Fuselage(
        xsecs=[
            FuselageXSec(xyz_c=[0, 0, 0], radius=0.5),
            FuselageXSec(xyz_c=[5, 0, 0], radius=0.5),
        ]
    )

    airplane = Airplane(wings=[wing], fuselages=[fuselage])

    assert len(airplane.wings) == 1
    assert len(airplane.fuselages) == 1


def test_airplane_custom_reference_values():
    """Test airplane with custom reference values."""
    wing = Wing(
        xsecs=[
            WingXSec(xyz_le=[0, 0, 0], chord=1.0),
            WingXSec(xyz_le=[0, 1, 0], chord=0.5),
        ]
    )

    s_ref_custom = 10.0
    c_ref_custom = 2.0
    b_ref_custom = 5.0

    airplane = Airplane(
        wings=[wing], s_ref=s_ref_custom, c_ref=c_ref_custom, b_ref=b_ref_custom
    )

    assert airplane.s_ref == s_ref_custom
    assert airplane.c_ref == c_ref_custom
    assert airplane.b_ref == b_ref_custom


def test_airplane_custom_xyz_ref():
    """Test airplane with custom reference point."""
    xyz_ref_custom = [1.5, 0.0, 0.2]

    wing = Wing(
        xsecs=[
            WingXSec(xyz_le=[0, 0, 0], chord=1.0),
            WingXSec(xyz_le=[0, 1, 0], chord=0.5),
        ]
    )

    airplane = Airplane(wings=[wing], xyz_ref=xyz_ref_custom)

    assert np.allclose(airplane.xyz_ref, xyz_ref_custom)


def test_airplane_analysis_specific_options():
    """Test airplane with analysis-specific options."""
    wing = Wing(
        xsecs=[
            WingXSec(xyz_le=[0, 0, 0], chord=1.0),
            WingXSec(xyz_le=[0, 1, 0], chord=0.5),
        ]
    )

    options = {asb.AeroBuildup: {"include_wave_drag": False}}

    airplane = Airplane(wings=[wing], analysis_specific_options=options)

    assert asb.AeroBuildup in airplane.analysis_specific_options
    assert (
        airplane.analysis_specific_options[asb.AeroBuildup]["include_wave_drag"]
        == False
    )


def test_airplane_no_wings_no_fuselages_raises_error():
    """Test that airplane with no wings or fuselages raises error for reference values."""
    with pytest.raises(ValueError):
        Airplane()  ### Should raise ValueError for missing s_ref


def test_airplane_fuselage_only_reference_values():
    """Test that airplane with only fuselage uses fuselage for reference values."""
    fuselage = Fuselage(
        xsecs=[
            FuselageXSec(xyz_c=[0, 0, 0], radius=0.5),
            FuselageXSec(xyz_c=[5, 0, 0], radius=0.5),
        ]
    )

    airplane = Airplane(fuselages=[fuselage])

    ### Should use fuselage properties for reference
    assert airplane.s_ref > 0
    assert airplane.c_ref > 0


def test_airplane_with_propulsors():
    """Test airplane with propulsors."""
    wing = Wing(
        xsecs=[
            WingXSec(xyz_le=[0, 0, 0], chord=1.0),
            WingXSec(xyz_le=[0, 1, 0], chord=0.5),
        ]
    )
    propulsor = Propulsor(xyz_c=[0.5, 0, 0], radius=0.1)

    airplane = Airplane(wings=[wing], propulsors=[propulsor])

    assert len(airplane.propulsors) == 1


def test_airplane_wing_span_computation():
    """Test that wing span is correctly used for b_ref."""
    wing = Wing(
        symmetric=False,  ### Asymmetric wing
        xsecs=[
            WingXSec(xyz_le=[0, 0, 0], chord=1.0),
            WingXSec(xyz_le=[0, 4, 0], chord=1.0),
        ],
    )

    airplane = Airplane(wings=[wing])

    ### b_ref should match the wing span
    assert np.isclose(airplane.b_ref, wing.span())


def test_airplane_wing_area_computation():
    """Test that wing area is correctly used for s_ref."""
    wing = Wing(
        symmetric=False,
        xsecs=[
            WingXSec(xyz_le=[0, 0, 0], chord=1.0),
            WingXSec(xyz_le=[0, 2, 0], chord=1.0),
        ],
    )

    airplane = Airplane(wings=[wing])

    ### Should match wing area
    assert np.isclose(airplane.s_ref, wing.area())


def test_airplane_mac_computation():
    """Test that mean aerodynamic chord is correctly computed."""
    wing = Wing(
        xsecs=[
            WingXSec(xyz_le=[0, 0, 0], chord=2.0),
            WingXSec(xyz_le=[0, 4, 0], chord=1.0),
        ]
    )

    airplane = Airplane(wings=[wing])

    ### Should match wing MAC
    assert np.isclose(airplane.c_ref, wing.mean_aerodynamic_chord())


def test_airplane_empty_analysis_options():
    """Test airplane with empty analysis options."""
    wing = Wing(
        xsecs=[
            WingXSec(xyz_le=[0, 0, 0], chord=1.0),
            WingXSec(xyz_le=[0, 1, 0], chord=0.5),
        ]
    )

    airplane = Airplane(wings=[wing], analysis_specific_options={})

    assert airplane.analysis_specific_options == {}


def test_airplane_multiple_analysis_options():
    """Test airplane with multiple analysis-specific options."""
    wing = Wing(
        xsecs=[
            WingXSec(xyz_le=[0, 0, 0], chord=1.0),
            WingXSec(xyz_le=[0, 1, 0], chord=0.5),
        ]
    )

    options = {
        asb.AeroBuildup: {"include_wave_drag": True, "fuselage_lift_carry_over": 0.8},
        "custom_analysis": {"param1": 42},
    }

    airplane = Airplane(wings=[wing], analysis_specific_options=options)

    assert len(airplane.analysis_specific_options) == 2
    assert (
        airplane.analysis_specific_options[asb.AeroBuildup]["include_wave_drag"] == True
    )


def test_airplane_realistic_configuration():
    """Test a realistic airplane configuration."""
    ### Main wing
    main_wing = Wing(
        name="Main Wing",
        xsecs=[
            WingXSec(xyz_le=[0, 0, 0], chord=1.5),
            WingXSec(xyz_le=[0.2, 5, 0], chord=0.8),
        ],
    )

    ### Horizontal stabilizer
    h_stab = Wing(
        name="H-Stab",
        xsecs=[
            WingXSec(xyz_le=[8, 0, 0], chord=0.6),
            WingXSec(xyz_le=[8.1, 1.5, 0], chord=0.4),
        ],
    )

    ### Fuselage
    fuselage = Fuselage(
        name="Fuselage",
        xsecs=[
            FuselageXSec(xyz_c=[0, 0, 0], radius=0.3),
            FuselageXSec(xyz_c=[5, 0, 0], radius=0.4),
            FuselageXSec(xyz_c=[10, 0, 0], radius=0.2),
        ],
    )

    airplane = Airplane(
        name="Conventional Aircraft",
        wings=[main_wing, h_stab],
        fuselages=[fuselage],
        xyz_ref=[5, 0, 0],
    )

    assert airplane.name == "Conventional Aircraft"
    assert len(airplane.wings) == 2
    assert len(airplane.fuselages) == 1
    assert airplane.s_ref > 0
    assert airplane.c_ref > 0
    assert airplane.b_ref > 0


def test_airplane_symmetric_wing():
    """Test airplane with symmetric wing."""
    wing = Wing(
        symmetric=True,
        xsecs=[
            WingXSec(xyz_le=[0, 0, 0], chord=1.0),
            WingXSec(xyz_le=[0, 2, 0], chord=0.5),
        ],
    )

    airplane = Airplane(wings=[wing])

    ### Symmetric wing should have doubled area
    expected_area = wing.area()
    assert np.isclose(airplane.s_ref, expected_area)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
