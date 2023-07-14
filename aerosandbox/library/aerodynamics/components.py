import aerosandbox.numpy as np
from typing import Union


def CDA_control_linkage(
        Re_l: Union[float, np.ndarray],
        linkage_length: Union[float, np.ndarray],
        is_covered: Union[bool, np.ndarray] = False,
        is_top: Union[bool, np.ndarray] = False,
) -> Union[float, np.ndarray]:
    """
    Computes the drag coefficient of a typical control usage as used on a well-manufactured RC airplane.

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
        local_chord,
        control_surface_span,
        local_thickness_over_chord=0.12,
        control_surface_hinge_x=0.75,
        n_side_gaps=2,
        side_gap_width=None,
        hinge_gap_width=None,
):
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


def CDA_bolts_and_rivets(
        diameter: float,
        kind: str = "flush_rivet"
):
    S_ref = np.pi * diameter ** 2 / 4

    if kind == "flush_rivet":
        CDA = 0.002 * S_ref
    elif kind == "round_rivet":
        CDA = 0.04 * S_ref
    elif kind == "flat_head_bolt":
        CDA == 0.02 * S_ref
    elif kind == "round_head_bolt":
        CDA == 0.32 * S_ref
    elif kind == "cylindrical_bolt":
        CDA == 0.42 * S_ref
    elif kind == "hex_bolt":
        CDA == 0.80 * S_ref
    else:
        raise ValueError("Invalid `kind` of bolt or rivet.")

    return CDA


def CDA_pitot_static_tube():
    pass
