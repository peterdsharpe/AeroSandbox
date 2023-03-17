import aerosandbox as asb
import aerosandbox.numpy as np
import aerosandbox.tools.units as u
from .raymer_fudge_factors import advanced_composites
from typing import Union


# From Raymer, Aircraft Design: A Conceptual Approach, 5th Ed.
# Section 15.3.2: Cargo/Transport Weights

def mass_wing(
        wing: asb.Wing,
        design_mass_TOGW: float,
        ultimate_load_factor: float,
        use_advanced_composites: bool = False,
) -> float:
    """
    Computes the mass of the wing for a cargo/transport aircraft, according to Raymer's Aircraft Design: A Conceptual
    Approach.

    Note: Torenbeek's wing mass model is likely more accurate; see `mass_wing()` in `torenbeek_weights.py` (same
    directory).

    Args:
        wing: The wing object.

        design_mass_TOGW: The design take-off gross weight of the entire airplane [kg].

        ultimate_load_factor: Ultimate load factor of the airplane.

        use_advanced_composites: Whether to use advanced composites for the wing. If True, the wing mass is modified
        accordingly.

    Returns:
        Wing mass [kg].
    """
    airfoil_thicknesses = [
        xsec.airfoil.max_thickness()
        for xsec in wing.xsecs
    ]

    airfoil_t_over_c = np.min(airfoil_thicknesses)

    return (
            0.0051 *
            (design_mass_TOGW / u.lbm * ultimate_load_factor) ** 0.557 *
            (wing.area('planform') / u.foot ** 2) ** 0.649 *
            wing.aspect_ratio() ** 0.5 *
            airfoil_t_over_c ** -0.4 *
            (1 + wing.taper_ratio()) ** 0.1 *
            np.cosd(wing.mean_sweep_angle()) ** -1 *
            (wing.control_surface_area() / u.foot ** 2) ** 0.1 *
            (advanced_composites["wing"] if use_advanced_composites else 1)
    ) * u.lbm


def mass_hstab(
        hstab: asb.Wing,
        design_mass_TOGW: float,
        ultimate_load_factor: float,
        wing_to_hstab_distance: float,
        fuselage_width_at_hstab_intersection: float,
        aircraft_y_radius_of_gyration: float = None,
        use_advanced_composites: bool = False,
) -> float:
    """
    Computes the mass of the horizontal stabilizer for a cargo/transport aircraft, according to Raymer's Aircraft
    Design: A Conceptual Approach.

    Args:
        hstab: The horizontal stabilizer object.

        design_mass_TOGW: The design take-off gross weight of the entire airplane [kg].

        ultimate_load_factor: Ultimate load factor of the airplane.

        wing_to_hstab_distance: Distance from the wing's root-quarter-chord-point to the hstab's
        root-quarter-chord-point [m].

        fuselage_width_at_hstab_intersection: Width of the fuselage at the intersection of the wing and hstab [m].

        aircraft_y_radius_of_gyration: Radius of gyration of the aircraft about the y-axis [m]. If None, estimates
        this as `0.3 * wing_to_hstab_distance`.

        use_advanced_composites: Whether to use advanced composites for the hstab. If True, the hstab mass is modified
        accordingly.

    Returns:
        The mass of the horizontal stabilizer [kg].
    """
    if aircraft_y_radius_of_gyration is None:
        aircraft_y_radius_of_gyration = 0.3 * wing_to_hstab_distance

    area = hstab.area()

    ### Determine if the hstab is all-moving or not
    all_moving = True
    for xsec in hstab.xsecs:
        for control_surface in xsec.control_surfaces:
            if (
                    (control_surface.trailing_edge and control_surface.hinge_point > 0) or
                    (not control_surface.trailing_edge and control_surface.hinge_point < 1)
            ):
                all_moving = False
                break

    return (
            0.0379 *
            (1.143 if all_moving else 1) *
            (1 + fuselage_width_at_hstab_intersection / hstab.span()) ** -0.25 *
            (design_mass_TOGW / u.lbm) ** 0.639 *
            ultimate_load_factor ** 0.10 *
            (area / u.foot ** 2) ** 0.75 *
            (wing_to_hstab_distance / u.foot) ** -1 *
            (aircraft_y_radius_of_gyration / u.foot) ** 0.704 *
            np.cosd(hstab.mean_sweep_angle()) ** -1 *
            hstab.aspect_ratio() ** 0.166 *
            (1 + hstab.control_surface_area() / area) ** 0.1 *
            (advanced_composites["tails"] if use_advanced_composites else 1)
    ) * u.lbm


def mass_vstab(
        vstab: asb.Wing,
        design_mass_TOGW: float,
        ultimate_load_factor: float,
        wing_to_vstab_distance: float,
        is_t_tail: bool = False,
        aircraft_z_radius_of_gyration: float = None,
        use_advanced_composites: bool = False,
) -> float:
    """
    Computes the mass of the vertical stabilizer for a cargo/transport aircraft, according to Raymer's Aircraft
    Design: A Conceptual Approach.

    Args:
        vstab: The vertical stabilizer object.

        design_mass_TOGW: The design take-off gross weight of the entire airplane [kg].

        ultimate_load_factor: Ultimate load factor of the airplane.

        wing_to_vstab_distance: Distance from the wing's root-quarter-chord-point to the vstab's
        root-quarter-chord-point [m].

        is_t_tail: Whether the airplane is a T-tail or not.

        aircraft_z_radius_of_gyration: The z-radius of gyration of the entire airplane [m]. If None, estimates this
        as `1 * wing_to_vstab_distance`.

        use_advanced_composites: Whether to use advanced composites for the vstab. If True, the vstab mass is modified
        accordingly.

    Returns:
        The mass of the vertical stabilizer [kg].

    """
    airfoil_thicknesses = [
        xsec.airfoil.max_thickness()
        for xsec in vstab.xsecs
    ]

    airfoil_t_over_c = np.min(airfoil_thicknesses)

    if aircraft_z_radius_of_gyration is None:
        aircraft_z_radius_of_gyration = 1 * wing_to_vstab_distance

    return (
            0.0026 *
            (1 + (1 if is_t_tail else 0)) ** 0.225 *
            (design_mass_TOGW / u.lbm) ** 0.556 *
            ultimate_load_factor ** 0.536 *
            (wing_to_vstab_distance / u.foot) ** -0.5 *
            (vstab.area('planform') / u.foot ** 2) ** 0.5 *
            (aircraft_z_radius_of_gyration / u.foot) ** 0.875 *
            np.cosd(vstab.mean_sweep_angle()) ** -1 *
            vstab.aspect_ratio() ** 0.35 *
            airfoil_t_over_c ** -0.5 *
            (advanced_composites["tails"] if use_advanced_composites else 1)
    ) * u.lbm


def mass_fuselage(
        fuselage: asb.Fuselage,
        design_mass_TOGW: float,
        ultimate_load_factor: float,
        L_over_D: float,
        main_wing: asb.Wing,
        n_cargo_doors: int = 1,
        has_aft_clamshell_door: bool = False,
        landing_gear_mounted_on_fuselage: bool = False,
        use_advanced_composites: bool = False,
) -> float:
    """
    Computes the mass of the fuselage for a cargo/transport aircraft, according to Raymer's Aircraft Design: A
    Conceptual Approach.

    Args:
        fuselage: The fuselage object.

        design_mass_TOGW: The design take-off gross weight of the entire airplane [kg].

        ultimate_load_factor: Ultimate load factor of the airplane.

        L_over_D: The lift-to-drag ratio of the airplane in cruise.

        main_wing: The main wing object. Can be:

            * An instance of an AeroSandbox wing object (`asb.Wing`)

            * None, if the airplane has no main wing.

        n_cargo_doors: The number of cargo doors on the fuselage.

        has_aft_clamshell_door: Whether or not the fuselage has an aft clamshell door.

        landing_gear_mounted_on_fuselage: Whether or not the landing gear is mounted on the fuselage.

        use_advanced_composites: Whether to use advanced composites for the fuselage. If True, the fuselage mass is
        modified accordingly.

    Returns:
        The mass of the fuselage [kg].
    """
    K_door = (1 + (0.06 * n_cargo_doors)) * (1.12 if has_aft_clamshell_door else 1)

    K_lg = 1.12 if landing_gear_mounted_on_fuselage else 1

    fuselage_structural_length = fuselage.length()

    if main_wing is not None:
        K_ws = (
                0.75 *
                (
                        (1 + 2 * main_wing.taper_ratio()) /
                        (1 + main_wing.taper_ratio())
                ) *
                (
                        main_wing.span() / fuselage_structural_length *
                        np.tand(main_wing.mean_sweep_angle())
                )
        )
    else:
        K_ws = 0

    return (
            0.3280 *
            K_door *
            K_lg *
            (design_mass_TOGW / u.lbm * ultimate_load_factor) ** 0.5 *
            (fuselage_structural_length / u.foot) ** 0.25 *
            (fuselage.area_wetted() / u.foot ** 2) ** 0.302 *
            (1 + K_ws) ** 0.04 *
            L_over_D ** 0.10 * # L/D
            (advanced_composites["fuselage/nacelle"] if use_advanced_composites else 1)
    ) * u.lbm


def mass_main_landing_gear(
        main_gear_length: float,
        landing_speed: float,
        design_mass_TOGW: float,
        is_kneeling: bool = False,
        n_gear: int = 2,
        n_wheels: int = 12,
        n_shock_struts: int = 4,
        use_advanced_composites: bool = False,
) -> float:
    """
    Computes the mass of the main landing gear for a cargo/transport aircraft, according to Raymer's Aircraft Design:
    A Conceptual Approach.

    Args:
        main_gear_length: length of the main landing gear [m].

        landing_speed: landing speed [m/s].

        design_mass_TOGW: The design take-off gross weight of the entire airplane [kg].

        is_kneeling: whether the main landing gear is capable of kneeling.

        n_gear: number of landing gear.

        n_wheels: number of wheels in total on the main landing gear.

        n_shock_struts: number of shock struts.

        use_advanced_composites: Whether to use advanced composites for the landing gear. If True, the landing gear mass
        is modified accordingly.

    Returns:
        mass of the main landing gear [kg].
    """

    K_mp = 1.126 if is_kneeling else 1

    ultimate_landing_load_factor = n_gear * 1.5

    return (
            0.0106 *
            K_mp *  # non-kneeling LG
            (design_mass_TOGW / u.lbm) ** 0.888 *
            ultimate_landing_load_factor ** 0.25 *
            (main_gear_length / u.inch) ** 0.4 *
            n_wheels ** 0.321 *
            n_shock_struts ** -0.5 *
            (landing_speed / u.knot) ** 0.1 *
            (advanced_composites["landing_gear"] if use_advanced_composites else 1)
    ) * u.lbm


def mass_nose_landing_gear(
        nose_gear_length: float,
        design_mass_TOGW: float,
        is_kneeling: bool = False,
        n_gear: int = 1,
        n_wheels: int = 2,
        use_advanced_composites: bool = False,
) -> float:
    """
    Computes the mass of the nose landing gear for a cargo/transport aircraft, according to Raymer's Aircraft
    Design: A Conceptual Approach.

    Args:
        nose_gear_length: Length of nose landing gear when fully-extended [m].

        design_mass_TOGW: The design take-off gross weight of the entire airplane [kg].

        is_kneeling: Whether the nose landing gear is capable of kneeling.

        n_gear: Number of nose landing gear.

        n_wheels: Number of wheels in total on the nose landing gear.

        use_advanced_composites: Whether to use advanced composites for the landing gear. If True, the landing gear mass
        is modified accordingly.

    Returns:
        Mass of nose landing gear [kg].
    """
    K_np = 1.15 if is_kneeling else 1

    ultimate_landing_load_factor = n_gear * 1.5

    return (
            0.032 *
            K_np *
            (design_mass_TOGW / u.lbm) ** 0.646 *
            ultimate_landing_load_factor ** 0.2 *
            (nose_gear_length / u.inch) ** 0.5 *
            n_wheels ** 0.45 *
            (advanced_composites["landing_gear"] if use_advanced_composites else 1)
    ) * u.lbm


def mass_nacelles(
        nacelle_length: float,
        nacelle_width: float,
        nacelle_height: float,
        ultimate_load_factor: float,
        mass_per_engine: float,
        n_engines: int,
        is_pylon_mounted: bool = False,
        engines_have_propellers: bool = False,
        engines_have_thrust_reversers: bool = False,
        use_advanced_composites: bool = False,
) -> float:
    """
    Computes the mass of the nacelles for a cargo/transport aircraft, according to Raymer's Aircraft
    Design: A Conceptual Approach. Excludes the engine itself and immediate engine peripherals.

    Args:
        nacelle_length: length of the nacelle, front to back [m]

        nacelle_width: width of the nacelle [m]

        nacelle_height: height of the nacelle, top to bottom [m]

        ultimate_load_factor: ultimate load factor of the aircraft

        mass_per_engine: mass of the engine itself [kg]

        n_engines: number of engines

        is_pylon_mounted: whether the engine is pylon-mounted or not

        engines_have_propellers: whether the engines have propellers or not (e.g., a jet)

        engines_have_thrust_reversers: whether the engines have thrust reversers or not

        use_advanced_composites: Whether to use advanced composites for the nacelles. If True, the nacelles mass
        is modified accordingly.

    Returns:
        mass of the nacelles [kg]
    """
    K_ng = 1.017 if is_pylon_mounted else 1

    K_p = 1.4 if engines_have_propellers else 1
    K_tr = 1.18 if engines_have_thrust_reversers else 1

    mass_per_engine_with_contents = np.softmax(
        (2.331 * (mass_per_engine / u.lbm) ** 0.901) * K_p * K_tr * u.lbm,
        mass_per_engine,
        hardness=10 / mass_per_engine
    )

    nacelle_wetted_area = (
            nacelle_length * nacelle_height * 2 +
            nacelle_width * nacelle_height * 2
    )

    return (
            0.6724 *
            K_ng *
            (nacelle_length / u.foot) ** 0.10 *
            (nacelle_width / u.foot) ** 0.294 *
            (ultimate_load_factor) ** 0.119 *
            (mass_per_engine_with_contents / u.lbm) ** 0.611 *
            (n_engines) ** 0.984 *
            (nacelle_wetted_area / u.foot ** 2) ** 0.224 *
            (advanced_composites["fuselage/nacelle"] if use_advanced_composites else 1)
    )


def mass_engine_controls(
        n_engines: int,
        cockpit_to_engine_length: float,
) -> float:
    """
    Computes the mass of the engine controls for a cargo/transport aircraft, according to Raymer's Aircraft
    Design: A Conceptual Approach.

    Args:
        n_engines: The number of engines in the aircraft.

        cockpit_to_engine_length: The distance from the cockpit to the engine [m].

    Returns:
        The mass of the engine controls [kg].
    """
    return (
            5 * n_engines +
            0.80 * (cockpit_to_engine_length / u.foot) * n_engines
    ) * u.lbm


def mass_starter(
        n_engines: int,
        mass_per_engine: float,
) -> float:
    """
    Computes the mass of the engine starter for a cargo/transport aircraft, according to Raymer's Aircraft
    Design: A Conceptual Approach.

    Args:
        n_engines: The number of engines in the aircraft.

        mass_per_engine: The mass of the engine [kg].

    Returns:
        The mass of the engine starter [kg].
    """
    return (
            49.19 * (
            mass_per_engine / u.lbm * n_engines
            / 1000
    ) ** 0.541
    ) * u.lbm


def mass_fuel_system(
        fuel_volume: float,
        n_tanks: int,
        fraction_in_integral_tanks: float = 0.5,
) -> float:
    """
    Computes the mass of the fuel system (e.g., tanks, pumps, but not the fuel itself) for a cargo/transport
    aircraft, according to Raymer's Aircraft Design: A Conceptual Approach.

    Args:
        fuel_volume: The volume of fuel in the aircraft [m^3].

        n_tanks: The number of fuel tanks in the aircraft.

        fraction_in_integral_tanks: The fraction of the fuel volume that is in integral tanks, as opposed to
        protected tanks.

    Returns:
        The mass of the fuel system [kg].
    """

    fraction_in_protected_tanks = 1 - fraction_in_integral_tanks
    return (
            2.405 *
            (fuel_volume / u.gallon) ** 0.606 *
            (1 + fraction_in_integral_tanks) ** -1 *
            (1 + fraction_in_protected_tanks) *
            n_tanks ** 0.5
    ) * u.lbm


def mass_flight_controls(
        airplane: asb.Airplane,
        aircraft_Iyy: float,
        fraction_of_mechanical_controls: int = 0,
) -> float:
    """
    Computes the added mass of the flight control surfaces (and any applicable linkages, in the case of mechanical
    controls) for a cargo/transport aircraft, according to Raymer's Aircraft Design: A Conceptual Approach.

    Args:
        airplane: The airplane to calculate the mass of the flight controls for.

        aircraft_Iyy: The moment of inertia of the aircraft about the y-axis.

        fraction_of_mechanical_controls: The fraction of the flight controls that are mechanical, as opposed to
        hydraulic.

    Returns:
        The mass of the flight controls [kg].
    """

    ### Compute how many functions the control surfaces are performing (e.g., aileron, elevator, flap, rudder, etc.)
    N_functions_performed_by_controls = 0
    for wing in airplane.wings:
        N_functions_performed_by_controls += len(wing.get_control_surface_names())

    ### Compute the control surface area
    control_surface_area = 0
    for wing in airplane.wings:
        control_surface_area += wing.control_surface_area()

    return (
            145.9 *
            N_functions_performed_by_controls ** 0.554 *  # number of functions performed by controls
            (1 + fraction_of_mechanical_controls) ** -1 *
            (control_surface_area / u.foot ** 2) ** 0.20 *
            (aircraft_Iyy / (u.lbm * u.foot ** 2) * 1e-6) ** 0.07
    ) * u.lbm


def mass_APU(
        mass_APU_uninstalled: float,
):
    """
    Computes the mass of the auxiliary power unit (APU) for a cargo/transport aircraft, according to Raymer's Aircraft
    Design: A Conceptual Approach.

    Args:
        mass_APU_uninstalled: The mass of the APU uninstalled [kg].

    Returns:
        The mass of the APU, as installed [kg].
    """
    return 2.2 * mass_APU_uninstalled


def mass_instruments(
        fuselage: asb.Fuselage,
        main_wing: asb.Wing,
        n_engines: int,
        n_crew: Union[int, float],
        engine_is_reciprocating: bool = False,
        engine_is_turboprop: bool = False,
):
    """
    Computes the mass of the flight instruments for a cargo/transport aircraft, according to Raymer's Aircraft
    Design: A Conceptual Approach.

    Args:
        fuselage: The fuselage of the airplane.

        main_wing: The main wing of the airplane.

        n_engines: The number of engines on the airplane.

        n_crew: The number of crew members on the airplane. Use 0.5 for a UAV.

        engine_is_reciprocating: Whether the engine is reciprocating.

        engine_is_turboprop: Whether the engine is a turboprop.

    Returns:
        The mass of the instruments [kg]
    """
    K_r = 1.133 if engine_is_reciprocating else 1

    K_tp = 0.793 if engine_is_turboprop else 1

    return (
            4.509 *
            K_r *
            K_tp *
            n_crew ** 0.541 *
            n_engines *
            (fuselage.length() / u.foot * main_wing.span() / u.foot) ** 0.5
    ) * u.lbm


def mass_hydraulics(
        airplane: asb.Airplane,
        fuselage: asb.Fuselage,
        main_wing: asb.Wing,
):
    """
    Computes the mass of the hydraulic system for a cargo/transport aircraft, according to Raymer's Aircraft
    Design: A Conceptual Approach.

    Args:
        airplane: The airplane to calculate the mass of the hydraulic system for.

        fuselage: The fuselage of the airplane.

        main_wing: The main wing of the airplane.

    Returns:
        The mass of the hydraulic system [kg].
    """
    N_functions_performed_by_controls = 0
    for wing in airplane.wings:
        N_functions_performed_by_controls += len(wing.get_control_surface_names())

    return (
            0.2673 *
            N_functions_performed_by_controls *
            (fuselage.length() / u.foot * main_wing.span() / u.foot) ** 0.937
    ) * u.lbm


def mass_electrical(
        system_electrical_power_rating: float,
        electrical_routing_distance: float,
        n_engines: int,
):
    """
    Computes the mass of the electrical system for a cargo/transport aircraft, according to Raymer's Aircraft
    Design: A Conceptual Approach.

    Args:


        system_electrical_power_rating: The total electrical power rating of the aircraft's electrical system [Watts].

            Typical values:
                * Transport airplane: 40,000 - 60,000 W
                * Fighter/bomber airplane: 110,000 - 160,000 W

        electrical_routing_distance: The electrical routing distance, generators to avionics to cockpit. [meters]

    Returns:

        The mass of the electrical system [kg].

    """

    return (
            7.291 *
            (system_electrical_power_rating / 1e3) ** 0.782 *
            (electrical_routing_distance / u.foot) ** 0.346 *
            (n_engines) ** 0.10
    ) * u.lbm


def mass_avionics(
        mass_uninstalled_avionics: float,
):
    """
    Computes the mass of the avionics for a cargo/transport aircraft, according to Raymer's Aircraft
    Design: A Conceptual Approach.

    Args:
        mass_uninstalled_avionics: The mass of the avionics, before installation [kg].

    Returns:
        The mass of the avionics, as installed [kg].
    """
    return (
            1.73 *
            (mass_uninstalled_avionics / u.lbm) ** 0.983
    ) * u.lbm


def mass_furnishings(
        n_crew: Union[int, float],
        mass_cargo: float,
        fuselage: asb.Fuselage,
):
    """
    Computes the mass of the furnishings for a cargo/transport aircraft, according to Raymer's Aircraft
    Design: A Conceptual Approach. Does not include cargo handling gear or seats.

    Args:
        n_crew: The number of crew members on the airplane. Use 0.5 for a UAV.

        mass_cargo: The mass of the cargo [kg].

        fuselage: The fuselage of the airplane.

    Returns:
        The mass of the furnishings [kg].
    """
    return (
            0.0577 *
            n_crew ** 0.1 *
            (mass_cargo / u.lbm) ** 0.393 *
            (fuselage.area_wetted() / u.foot ** 2) ** 0.75
    ) * u.lbm


def mass_air_conditioning(
        n_crew: int,
        n_pax: int,
        volume_pressurized: float,
        mass_uninstalled_avionics: float,
):
    """
    Computes the mass of the air conditioning system for a cargo/transport aircraft, according to Raymer's Aircraft
    Design: A Conceptual Approach.

    Args:
        n_crew: The number of crew members on the airplane.

        n_pax: The number of passengers on the airplane.

        volume_pressurized: The volume of the pressurized cabin [meters^3].

        mass_uninstalled_avionics: The mass of the avionics, before installation [kg].

    Returns:
        The mass of the air conditioning system [kg].
    """
    return (
            62.36 *
            (n_crew + n_pax) ** 0.25 *
            (volume_pressurized / u.foot ** 3 / 1e3) ** 0.604 *
            (mass_uninstalled_avionics / u.lbm) ** 0.10
    ) * u.lbm


def mass_anti_ice(
        design_mass_TOGW: float,
):
    """
    Computes the mass of the anti-ice system for a cargo/transport aircraft, according to Raymer's Aircraft
    Design: A Conceptual Approach.

    Args:
        design_mass_TOGW: The design takeoff gross weight of the entire airplane [kg].

    Returns:
        The mass of the anti-ice system [kg].
    """
    return 0.002 * design_mass_TOGW


def mass_handling_gear(
        design_mass_TOGW: float,
):
    """
    Computes the mass of the handling gear for a cargo/transport aircraft, according to Raymer's Aircraft
    Design: A Conceptual Approach.

    Args:
        design_mass_TOGW: The design takeoff gross weight of the entire airplane [kg].

    Returns:
        The mass of the handling gear [kg].
    """
    return 3e-4 * design_mass_TOGW


def mass_military_cargo_handling_system(
        cargo_floor_area: float,
):
    """
    Computes the mass of the military cargo handling system for a cargo/transport aircraft, according to Raymer's
    Aircraft Design: A Conceptual Approach.

    Args:
        cargo_floor_area: The floor area of the cargo compartment [meters^2].

    Returns:
        The mass of the military cargo handling system [kg].
    """
    return (
            2.4 *
            (cargo_floor_area / u.foot ** 2)
    ) * u.lbm
