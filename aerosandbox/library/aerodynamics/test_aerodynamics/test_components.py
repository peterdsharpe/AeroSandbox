import pytest

from aerosandbox.library.aerodynamics.components import (
    CDA_perpendicular_sheet_metal_joint,
)

ALL_DOCUMENTED_JOINT_KINDS = [
    "butt_joint_with_inside_joiner",
    "butt_joint_with_inside_weld",
    "butt_joint_with_outside_joiner",
    "butt_joint_with_outside_weld",
    "lap_joint_forward_facing_step",
    "lap_joint_backward_facing_step",
    "lap_joint_forward_facing_step_with_bevel",
    "lap_joint_backward_facing_step_with_bevel",
    "lap_joint_forward_facing_step_with_rounded_bevel",
    "lap_joint_backward_facing_step_with_rounded_bevel",
    "flush_lap_joint_forward_facing_step",
    "flush_lap_joint_backward_facing_step",
]


@pytest.mark.parametrize("kind", ALL_DOCUMENTED_JOINT_KINDS)
def test_all_documented_sheet_metal_joint_kinds_are_valid(kind):
    """
    Every `kind` listed in the docstring should be accepted.

    Notably, "lap_joint_forward_facing_step_with_rounded_bevel" used to raise
    ValueError due to a typo ('lap joint_...') in the internal lookup table.
    """
    CDA = CDA_perpendicular_sheet_metal_joint(
        joint_width=1.0,
        sheet_metal_thickness=0.001,
        kind=kind,
    )
    assert CDA > 0


def test_sheet_metal_joint_rounded_bevel_value():
    CDA = CDA_perpendicular_sheet_metal_joint(
        joint_width=2.0,
        sheet_metal_thickness=0.001,
        kind="lap_joint_forward_facing_step_with_rounded_bevel",
    )
    assert CDA == pytest.approx(0.04 * 2.0 * 0.001)


def test_sheet_metal_joint_invalid_kind_raises():
    with pytest.raises(ValueError):
        CDA_perpendicular_sheet_metal_joint(
            joint_width=1.0,
            sheet_metal_thickness=0.001,
            kind="not_a_joint",
        )


if __name__ == "__main__":
    pytest.main([__file__])
