import aerosandbox.numpy as np
from typing import Union


def CDA_control_linkage(
        Re_l: Union[float, np.ndarray],
        linkage_length: Union[float, np.ndarray],
        is_covered: Union[bool, np.ndarray] = False,
        is_top: Union[bool, np.ndarray] = False,
) -> Union[float, np.ndarray]:
    """
    Computes the drag area (CDA) of a typical control usage as used on a well-manufactured RC airplane.

    The drag area (CDA) is defined as: CDA == D / q, where:

        - D is the drag force (dimensionalized, e.g., in Newtons)

        - q is the freestream dynamic pressure (dimensionalized, e.g., in Pascals)

    See study with original data at `AeroSandbox/studies/LinkageDrag`.

    Data from:

        * Hepperle, Martin. "Drag of Linkages". https://www.mh-aerotools.de/airfoils/linkage.htm

        * Summarizes data from "Werner Würz, published in the papers of the ISF-Seminar in December 1989 in Baden, Switzerland."

    Args:

        Re_l: Reynolds number, with reference length as the length of the linkage.

        linkage_length: The length of the linkage. [m]

        is_covered: A boolean of whether an aerodynamic fairing is placed around the linkage.

        is_top: A boolean of whether the linkage is on the top surface of the wing (True) or the bottom surface (
            False). Differences in local boundary layer and inviscid effects cause local velocity changes.

    Returns: The drag area [m^2] of the control linkage.

    """
    x = dict(
        Re_l=Re_l,
        linkage_length=linkage_length,
        is_covered=is_covered,
        is_top=is_top
    )

    p = {
        'CD0'               : 7.833083680086374e-05,
        'CD1'               : 0.0001216877860785463,
        'c_length'          : 30.572471745477774,
        'covered_drag_ratio': 0.7520722978405192,
        'top_drag_ratio'    : 1.1139040832208857
    }

    Re = x["Re_l"]
    linkage_length = x["linkage_length"]
    is_covered = x["is_covered"]
    is_top = x["is_top"]

    side_drag_multiplier = np.where(
        is_top,
        p["top_drag_ratio"],
        1
    )
    covered_drag_multiplier = np.where(
        is_covered,
        p["covered_drag_ratio"],
        1
    )
    linkage_length_multiplier = 1 + p["c_length"] * linkage_length

    CDA_raw = (
            p["CD1"] / (Re / 1e5) +
            p["CD0"]
    )

    return side_drag_multiplier * covered_drag_multiplier * linkage_length_multiplier * CDA_raw


def CDA_control_surface_gaps(
        local_chord: float,
        control_surface_span: float,
        local_thickness_over_chord: float = 0.12,
        control_surface_hinge_x: float = 0.75,
        n_side_gaps: int = 2,
        side_gap_width: float = None,
        hinge_gap_width: float = None,
) -> float:
    """
    Computes the drag area (CDA) of the gaps associated with a typical wing control surface.
    (E.g., aileron, flap, elevator, rudder).

    The drag area (CDA) is defined as: CDA == D / q, where:

        - D is the drag force (dimensionalized, e.g., in Newtons)

        - q is the freestream dynamic pressure (dimensionalized, e.g., in Pascals)

    This drag area consists of two sources:

        1. Chordwise gaps at the side edges of the control surface ("side gaps")

        2. Spanwise gaps at the hinge line of the control surface ("hinge gap")

    Args:

        local_chord: The local chord of the wing at the midpoint of the control surface. [meters]

        control_surface_span: The span of the control surface. [meters]

        local_thickness_over_chord: The local thickness-to-chord ratio of the wing at the midpoint of the control
            surface. [nondimensional] For example, this is 0.12 for a NACA0012 airfoil.

        control_surface_hinge_x: The x-location of the hinge line of the control surface, as a fraction of the local
            chord. [nondimensional] Defaults to x_hinge / c = 0.75, which is typical for an aileron.

        n_side_gaps: The number of "side gaps" to count on this control surface when computing drag. Defaults to 2 (
            i.e., one inboard gap, one outboard gap), which is the simplest case of a wing with a single partial-span
            aileron. However, there may be cases where it is best to reduce this to 1 or 0. For example:

                * A wing with a single full-span aileron would have 1 side gap (at the wing root, but not at the tip).

                * A wing with a flap and aileron that share a chordwise gap would be best modeled by setting
                    n_side_gaps = 1 ( so that no double-counting occurs).

        side_gap_width: The width of the chordwise gaps at the side edges of the control surface [meters]. If this is
            left as the default (None), then a typical value will be computed based on the local chord and control surface
            span.

        hinge_gap_width: The width of the spanwise gap at the hinge line of the control surface [meters]. If this is
            left as the default (None), then a typical value will be computed based on the local chord.

    Returns: The drag area [m^2] of the gaps associated with the control surface. This should be added to the "clean"
        wing drag to get a more realistic drag estimate.

    """
    if side_gap_width is None:
        side_gap_width = np.maximum(
            np.maximum(
                0.002,
                0.006 * local_chord
            ),
            control_surface_span * 0.01
        )
    if hinge_gap_width is None:
        hinge_gap_width = 0.03 * local_chord

    ### Chordwise gaps (at side edges of control surface)
    """
    Based on Hoerner, "Fluid Dynamic Drag", 1965, p. 5-13. Figure 26, "Drag of longitudinal wing gaps, 
    tested on 2412 airfoil at C_L = 0.1 and Re_c = 2 * 10^6"
    """

    CDA_side_gaps = n_side_gaps * (side_gap_width * local_chord * local_thickness_over_chord) * 0.50

    ### Spanwise gaps (at hinge line of control surface)
    """
    Based on Hoerner, "Fluid Dynamic Drag", 1965, p. 5-13. Figure 27, "Evaluation of drag due to control gaps"
    """

    CDA_hinge_gap = 0.025 * hinge_gap_width * control_surface_span

    ### Total
    return CDA_side_gaps + CDA_hinge_gap


def CDA_protruding_bolt_or_rivet(
        diameter: float,
        kind: str = "flush_rivet"
):
    """
    Computes the drag area (CDA) of a protruding bolt or rivet.

    The drag area (CDA) is defined as: CDA == D / q, where:
        - D is the drag force (dimensionalized, e.g., in Newtons)
        - q is the freestream dynamic pressure (dimensionalized, e.g., in Pascals)

    Args:

        diameter: The diameter of the bolt or rivet. [meters]

        kind: The type of bolt or rivet. Valid options are:

            - "flush_rivet"

            - "round_rivet"

            - "flat_head_bolt"

            - "round_head_bolt"

            - "cylindrical_bolt"

            - "hex_bolt"

    Returns: The drag area [m^2] of the bolt or rivet.
    """
    S_ref = np.pi * diameter ** 2 / 4

    CD_factors = {
        "flush_rivet"     : 0.002,
        "round_rivet"     : 0.04,
        "flat_head_bolt"  : 0.02,
        "round_head_bolt" : 0.32,
        "cylindrical_bolt": 0.42,
        "hex_bolt"        : 0.80,
    }

    try:
        CDA = CD_factors[kind] * S_ref
    except KeyError:
        raise ValueError("Invalid `kind` of bolt or rivet.")

    return CDA


def CDA_perpendicular_sheet_metal_joint(
        joint_width: float,
        sheet_metal_thickness: float,
        kind: str = "butt_joint_with_inside_joiner"
):
    """
    Computes the drag area (CDA) of a sheet metal joint that is perpendicular to the flow.
        (E.g., spanwise on the wing, or circumferential on the fuselage).

    The drag area (CDA) is defined as: CDA == D / q, where:

        - D is the drag force (dimensionalized, e.g., in Newtons)

        - q is the freestream dynamic pressure (dimensionalized, e.g., in Pascals)

    Args:

        joint_width: The width of the joint (perpendicular to the airflow, e.g., spanwise on a wing). [meters]

        sheet_metal_thickness: The thickness of the sheet metal. [meters]

        kind: The type of joint. Valid options are:

            - "butt_joint_with_inside_joiner"

            - "butt_joint_with_inside_weld"

            - "butt_joint_with_outside_joiner"

            - "butt_joint_with_outside_weld"

            - "lap_joint_forward_facing_step"

            - "lap_joint_backward_facing_step"

            - "lap_joint_forward_facing_step_with_bevel"

            - "lap_joint_backward_facing_step_with_bevel"

            - "lap_joint_forward_facing_step_with_rounded_bevel"

            - "lap_joint_backward_facing_step_with_rounded_bevel"

            - "flush_lap_joint_forward_facing_step"

            - "flush_lap_joint_backward_facing_step"


    Returns: The drag area [m^2] of the sheet metal joint.

    """
    S_ref = joint_width * sheet_metal_thickness

    CD_factors = {
        "butt_joint_with_inside_joiner"                    : 0.01,
        "butt_joint_with_inside_weld"                      : 0.01,
        "butt_joint_with_outside_joiner"                   : 0.70,
        "butt_joint_with_outside_weld"                     : 0.51,
        "lap_joint_forward_facing_step"                    : 0.40,
        "lap_joint_backward_facing_step"                   : 0.22,
        "lap_joint_forward_facing_step_with_bevel"         : 0.11,
        "lap_joint_backward_facing_step_with_bevel"        : 0.24,
        "lap joint_forward_facing_step_with_rounded_bevel" : 0.04,
        "lap_joint_backward_facing_step_with_rounded_bevel": 0.16,
        "flush_lap_joint_forward_facing_step"              : 0.13,
        "flush_lap_joint_backward_facing_step"             : 0.07,
    }

    try:
        CDA = CD_factors[kind] * S_ref
    except KeyError:
        raise ValueError("Invalid `kind` of sheet metal joint.")

    return CDA


# def CDA_pitot_static_tube():
#     pass
#
#
# def CDA_landing_gear():
#     pass
