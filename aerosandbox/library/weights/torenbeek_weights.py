import aerosandbox as asb
import aerosandbox.numpy as np
import aerosandbox.tools.units as u
from typing import Dict, Union


# From Torenbeek: "Synthesis of Subsonic Airplane Design", 1976, Delft University Press
# Chapter 8: "Airplane Weight and Balance"

def mass_wing_simple(
        wing: asb.Wing,
        design_mass_TOGW: float,
        ultimate_load_factor: float,
        suspended_mass: float,
        main_gear_mounted_to_wing: bool = True,
) -> float:
    """
    Computes the mass of a wing of an aircraft, according to Torenbeek's "Synthesis of Subsonic 
    Airplane Design".

    This is the simple version of the wing weight model, which is found in:
    Section 8.4: Weight Prediction Data and Methods
    8.4.1: Airframe Structure
    Eq. 8-12

    A more detailed version of the wing weight model is available in the `mass_wing()` function in this same module.

    Args:

        wing: The wing object. Should be an AeroSandbox Wing object.

        design_mass_TOGW: The design takeoff gross weight of the entire aircraft [kg].

        ultimate_load_factor: The ultimate load factor of the aircraft. 1.5x the limit load factor.

        suspended_mass: The mass of the aircraft that is suspended from the wing [kg].

        main_gear_mounted_to_wing: Whether the main gear is mounted to the wing structure.

    Returns: The total mass of the wing [kg].

    """

    k_w = np.blend(
        (design_mass_TOGW - 5670) / 2000,
        6.67e-3,
        4.90e-3
    )

    span = wing.span() / np.cosd(wing.mean_sweep_angle(x_nondim=0.5))

    wing_root_thickness = wing.xsecs[0].airfoil.max_thickness() * wing.xsecs[0].chord

    return suspended_mass * (
            k_w *
            span ** 0.75 *
            (1 + (1.905 / span) ** 0.5) *
            ultimate_load_factor ** 0.55 *
            ((span / wing_root_thickness) / (suspended_mass / wing.area())) ** 0.30 *
            (1 if main_gear_mounted_to_wing else 0.95)
    )


def mass_wing_high_lift_devices(
        wing: asb.Wing,
        max_airspeed_for_flaps: float,
        flap_deflection_angle: float = 30,
        k_f1: float = 1.0,
        k_f2: float = 1.0
) -> float:
    """
    The function mass_high_lift() is designed to estimate the weight of the high-lift devices
        on an airplane wing. It uses Torenbeek's method, which is based on multiple factors
        like wing design and flap deflection.

    Args:

        wing, an instance of AeroSandbox's Wing class,

        max_airspeed_for_flaps, the maximum airspeed at which the flaps can be deployed [m/s]

        flap_deflection_angle, the angle to which the flaps can be deflected [deg]. Default value is 30 degrees.

        k_f1, configuration factor 1, with values:
                = 1.0  for single-slotted; double-slotted, fixed hinge
                = 1.15 for double: slotted, 4-bar movement; single-slotted Fowler
                = 1.3  for double-slotted Fowler
                = 1.45 for triple-slotted Fowler

        k_f2, configuration factor 2, with values:
                = 1.0  for slotted flaps with fixed vanes
                = 1.25 for double-slotted flaps with "variable geometry", i.e., extending
                           flaps with separately moving vanes or auxiliary flaps

    Returns: Mass of the wing's high-lift system only [kg]
    """
    # S_flaps represents the total area of the control surfaces (flaps) on the wing.
    S_flaps = wing.control_surface_area()

    # Wing span
    span = wing.span()

    # Sweep at 50% chord
    sweep_half_chord = wing.mean_sweep_angle(x_nondim=0.5)
    cos_sweep_half_chord = np.cosd(sweep_half_chord)

    # span_structural is the "structural" wing span, which takes into account the wing's sweep angle.
    span_structural = span / cos_sweep_half_chord

    # Airfoil thickness over chord ratio at root
    root_t_over_c = wing.xsecs[0].airfoil.max_thickness()

    # Torenbeek Eq. C-10
    k_f = k_f1 * k_f2

    mass_trailing_edge_flaps = S_flaps * (
            2.706 * k_f *
            (S_flaps * span_structural) ** (3 / 16) *
            (
                    (max_airspeed_for_flaps / 100) ** 2 *
                    np.sind(flap_deflection_angle) *
                    np.cosd(wing.mean_sweep_angle(x_nondim=1)) /
                    root_t_over_c
            ) ** (3 / 4)
    )

    mass_leading_edge_devices = 0

    mass_high_lift_devices = mass_trailing_edge_flaps + mass_leading_edge_devices

    return mass_high_lift_devices


def mass_wing_basic_structure(
        wing: asb.Wing,
        design_mass_TOGW: float,
        ultimate_load_factor: float,
        suspended_mass: float,
        never_exceed_airspeed: float,
        main_gear_mounted_to_wing: bool = True,
        strut_y_location: float = None,
        k_e: float = 0.95,
        return_dict: bool = False,
) -> Union[float, Dict[str, float]]:
    """
    Computes the mass of the basic structure of the wing of an aircraft, according to 
        Torenbeek's "Synthesis of Subsonic Airplane Design", 1976, Appendix C: "Prediction
        of Wing Structural Weight". This is the basic wing structure without movables like spoilers,
        high-lift devices, etc.

    Likely more accurate than the Raymer wing weight models.

    Args:

        wing: The wing object.

        design_mass_TOGW: The design takeoff gross weight of the entire aircraft [kg].

        ultimate_load_factor: The ultimate load factor of the aircraft [-]. 1.5x the limit load factor.

        suspended_mass: The mass of the aircraft that is suspended from the wing [kg]. It should exclude 
            any wing attachments that are not part of the wing structure.

        never_exceed_airspeed: The never-exceed airspeed of the aircraft [m/s]. Used for flutter calculations.

        main_gear_mounted_to_wing: Whether the main gear is mounted to the wing structure. Boolean.

        strut_y_location: The spanwise-location of the strut (if any), as measured from the wing root [meters]. If None,
            it is assumed that there is no strut (i.e., the wing is a cantilever beam).

        k_e: represents weight knockdowns due to bending moment relief from engines mounted in front of elastic axis.
            see Torenbeek unlabeled equations, between C-3 and C-4.
                k_e = 1.0 if engines are not wing mounted,
                k_e = 0.95 (default) two wing mounted engines in front of the elastic axis and 
                k_e = 0.90 four wing-mounted engines in front of the elastic axis

        return_dict: Whether to return a dictionary of all the intermediate values, or just the final mass. Defaults
            to False, which returns just the final mass [kg].

    Returns: If return_dict is False (default), returns a single value: the mass of the basic wing [kg]. If return_dict is
        True, returns a dictionary of all the intermediate values.

    """

    # Wing span
    span = wing.span()

    # Sweep at 50% chord
    sweep_half_chord = wing.mean_sweep_angle(x_nondim=0.5)
    cos_sweep_half_chord = np.cosd(sweep_half_chord)

    # Structural wing span
    span_structural = span / cos_sweep_half_chord

    # Airfoil thickness over chord ratio at root
    root_t_over_c = wing.xsecs[0].airfoil.max_thickness()

    # Torenbeek Eq. C-2
    # `k_no` represents penalties due to skin joints, non-tapered skin, minimum gauge, etc.
    k_no = 1 + (1.905 / span_structural) ** 0.5

    # Torenbeek Eq. C-3
    # `k_lambda` represents penalties due to taper ratio
    k_lambda = (1 + wing.taper_ratio()) ** 0.4

    # `k_uc` represents weight knockdowns due to undercarriage.
    k_uc = 1 if main_gear_mounted_to_wing else 0.95

    # Torenbeek Eq. C-4
    # `k_st` represents weight excrescence due to structural stiffness against flutter.
    k_st = (
            1 +
            9.06e-4 * (
                    (span * np.cosd(wing.mean_sweep_angle(x_nondim=0))) ** 3 /
                    design_mass_TOGW
            ) * (
                    never_exceed_airspeed / 100 / root_t_over_c
            ) ** 2 *
            cos_sweep_half_chord
    )

    # Torenbeek Eq. C-5
    # `k_b` represents weight knockdowns due to bending moment relief from strut location.
    if strut_y_location is None:
        k_b = 1
    else:
        k_b = 1 - (strut_y_location / (wing.span() / 2)) ** 2

    ### Use all the above to compute the basic wing structural mass
    mass_wing_basic = (
            4.58e-3 *
            k_no *
            k_lambda *
            k_e *
            k_uc *
            k_st *
            (
                    k_b * ultimate_load_factor * (0.8 * suspended_mass + 0.2 * design_mass_TOGW)
            ) ** 0.55 *
            span ** 1.675 *
            root_t_over_c ** -0.45 *
            cos_sweep_half_chord ** -1.325
    )

    if return_dict:
        return locals()
    else:
        return mass_wing_basic


def mass_wing_spoilers_and_speedbrakes(
        wing: asb.Wing,
        mass_basic_wing: float
) -> float:
    """
    The function mass_spoilers_and_speedbrakes() estimates the weight of the spoilers and speedbrakes
        according to Torenbeek's "Synthesis of Subsonic Airplane Design", 1976, Appendix C: "Prediction
        of Wing Structural Weight".

    N.B. the weight is coming out unrealistic and approx. 20-30% of the weight of the wing. This needs
        a correction. It uses normally the 12.2 kg/m^2 wing area.

    Args:

        wing: an instance of AeroSandbox's Wing class.

        mass_basic_wing: The basic weight of the wing (without spoilers, speedbrakes, flaps, slats) [kg]

    Returns: The mass of the spoilers and speed brakes only [kg]

    N.B. the weight estimation using the 12.2 kg/m^2 figure comes out too high if using
        the wing as a referenced area. Reduced to 1.5% of the basic wing mass.
    """
    # mass_spoilers_and_speedbrakes = np.softmax(
    #                                            12.2 * wing.area(),
    #                                            0.015 * mass_basic_wing
    #                                            )

    mass_spoilers_and_speedbrakes = 0.015 * mass_basic_wing

    return mass_spoilers_and_speedbrakes


def mass_wing(
        wing: asb.Wing,
        design_mass_TOGW: float,
        ultimate_load_factor: float,
        suspended_mass: float,
        never_exceed_airspeed: float,
        max_airspeed_for_flaps: float,
        main_gear_mounted_to_wing: bool = True,
        flap_deflection_angle: float = 30,
        strut_y_location: float = None,
        return_dict: bool = False,
) -> Union[float, Dict[str, float]]:
    """
    Computes the mass of a wing of an aircraft, according to Torenbeek's "Synthesis of Subsonic Airplane Design",
        1976, Appendix C: "Prediction of Wing Structural Weight".

    Likely more accurate than the Raymer wing weight models.

    Args:

        wing: The wing object.

        design_mass_TOGW: The design takeoff gross weight of the entire aircraft [kg].

        ultimate_load_factor: The ultimate load factor of the aircraft. 1.5x the limit load factor.

        suspended_mass: The mass of the aircraft that is suspended from the wing [kg].

        never_exceed_airspeed: The never-exceed airspeed of the aircraft [m/s]. Used for flutter calculations.

        max_airspeed_for_flaps: The maximum airspeed at which the flaps are allowed to be deployed [m/s]. In the
        absence of other information, 1.8x stall speed is a good guess.

        main_gear_mounted_to_wing: Whether the main gear is mounted to the wing structure.

        flap_deflection_angle: The maximum deflection angle of the flaps [deg].

        strut_y_location: The y-location of the strut (if any), relative to the wing's leading edge [m]. If None,
            it is assumed that there is no strut (i.e., the wing is a cantilever beam).

        return_dict: Whether to return a dictionary of all the intermediate values, or just the final mass. Defaults
            to False, which returns just the final mass.

    Returns: If return_dict is False (default), returns a single value: the total mass of the wing [kg]. If
        return_dict is True, returns a dictionary of all the intermediate values.

    """

    # High-lift mass estimation
    mass_high_lift_devices = mass_wing_high_lift_devices(
        wing=wing,
        max_airspeed_for_flaps=max_airspeed_for_flaps,
        flap_deflection_angle=flap_deflection_angle,
    )
    # Basic wing structure mass estimation
    mass_basic_wing = mass_wing_basic_structure(
        wing=wing,
        design_mass_TOGW=design_mass_TOGW,
        ultimate_load_factor=ultimate_load_factor,
        suspended_mass=suspended_mass,
        never_exceed_airspeed=never_exceed_airspeed,
        main_gear_mounted_to_wing=main_gear_mounted_to_wing,
        strut_y_location=strut_y_location,
    )
    # spoilers and speedbrake mass estimation
    mass_spoilers_speedbrakes = mass_wing_spoilers_and_speedbrakes(
        wing=wing,
        mass_basic_wing=mass_basic_wing
    )

    mass_wing_total = (
            mass_basic_wing +
            1.2 * (mass_high_lift_devices + mass_spoilers_speedbrakes)
    )

    if return_dict:
        return locals()
    else:
        return mass_wing_total


# def mass_hstab(
#         hstab: asb.Wing,
#         design_mass_TOGW: float,
#         ultimate_load_factor: float,
#         suspended_mass: float,
#         main_gear_mounted_to_wing: bool = True,
# ) -> float:
#
#     k_wt = 0.64

def mass_fuselage_simple(
        fuselage: asb.Fuselage,
        never_exceed_airspeed: float,
        wing_to_tail_distance: float,
):
    """
    Computes the mass of the fuselage, using Torenbeek's simple version of the calculation.

    Source:
    Torenbeek: "Synthesis of Subsonic Airplane Design", 1976
    Section 8.4: Weight Prediction Data and Methods
    8.4.1: Airframe Structure
    Eq. 8-16

    Args:

        fuselage: The fuselage object. Should be an AeroSandbox Fuselage object.

        never_exceed_airspeed: The never-exceed airspeed of the aircraft, in m/s.

        wing_to_tail_distance: The distance from the quarter-chord of the wing to the quarter-chord of the tail,
        in meters.

    Returns: The mass of the fuselage, in kg.

    """
    widths = [
        xsec.width
        for xsec in fuselage.xsecs
    ]

    max_width = np.softmax(
        *widths,
        softness=np.mean(np.array(widths)) * 0.01
    )

    heights = [
        xsec.height
        for xsec in fuselage.xsecs
    ]

    max_height = np.softmax(
        *heights,
        softness=np.mean(np.array(heights)) * 0.01
    )

    return (
            0.23 *
            (
                    never_exceed_airspeed *
                    wing_to_tail_distance /
                    (max_width + max_height)
            ) ** 0.5 *
            fuselage.area_wetted() ** 1.2
    )


def mass_fuselage(
        fuselage: asb.Fuselage,
        design_mass_TOGW: float,
        ultimate_load_factor: float,
        never_exceed_airspeed: float,
        wing_to_tail_distance: float,
):
    # TODO Torenbeek Appendix D (PDF page 477)

    # Stage 1: Calculate the weight of the fuselage shell, which carries the primary loads and contributes
    # approximately 1/3 to 1/2 of the fuselage weight ("gross shell weight").

    # Torenbeek Eq. D-3
    fuselage.fineness_ratio()

    fuselage_quasi_slenderness_ratio = fuselage.fineness_ratio(assumed_shape="sears_haack")

    k_lambda = np.softmin(
        0.56 * fuselage.fineness_ratio(assumed_shape="sears_haack")
    )

    W_sk = 0.05428 * k_lambda * S_g ** 1.07 * never_exceed_airspeed ** 0.743

    W_g = W_sk + W_str + W_fr


def mass_propeller(
        propeller_diameter: float,
        propeller_power: float,
        n_blades: int,
) -> float:
    """
    Computes the mass of a propeller.

    From Torenbeek: "Synthesis of Subsonic Airplane Design", 1976, Delft University Press.
    Table 8-9 (pg. 286, PDF page 306)

    Args:

        propeller_diameter: Propeller diameter, in meters.

        propeller_power: Propeller power, in watts.

        n_blades: Number of propeller blades.

    Returns: Propeller mass, in kilograms.

    """
    return (
            0.108 *
            n_blades *
            (
                    (propeller_diameter / u.foot) *
                    (propeller_power / u.horsepower)
            ) ** 0.78174
    ) * u.lbm
