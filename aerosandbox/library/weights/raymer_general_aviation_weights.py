import aerosandbox as asb
import aerosandbox.numpy as np
import aerosandbox.tools.units as u
from .raymer_fudge_factors import advanced_composites


# From Raymer: "Aircraft Design: A Conceptual Approach", 5th Ed.
# Section 15.3.3: General Aviation Weights

def mass_wing(
        wing: asb.Wing,
        design_mass_TOGW: float,
        ultimate_load_factor: float,
        mass_fuel_in_wing: float,
        cruise_op_point: asb.OperatingPoint,
        use_advanced_composites: bool = False,
) -> float:
    """
    Computes the mass of a wing of a general aviation aircraft, according to Raymer's Aircraft Design: A Conceptual
    Approach.

    Note: Torenbeek's wing mass model is likely more accurate; see `mass_wing()` in `torenbeek_weights.py` (same
    directory).

    Args:
        wing: The wing object.

        design_mass_TOGW: The design takeoff gross weight of the entire aircraft [kg].

        ultimate_load_factor: The ultimate load factor of the aircraft.

        mass_fuel_in_wing: The mass of fuel in the wing [kg]. If there is no fuel in the wing, set this to 0.

            Note: Model extrapolates strangely for infinitesimally-small-but-nonzero fuel masses; don't let an
            optimizer land here.

        cruise_op_point: The cruise operating point of the aircraft.

        use_advanced_composites: Whether to use advanced composites for the wing. If True, the wing mass is modified
        accordingly.

    Returns: The mass of the wing [kg].

    """
    try:
        fuel_is_in_wing = bool(mass_fuel_in_wing > 0)
    except RuntimeError:
        fuel_is_in_wing = True

    if fuel_is_in_wing:
        fuel_weight_factor = np.softmax(
            (mass_fuel_in_wing / u.lbm) ** 0.0035,
            1,
            hardness=1000
        )
    else:
        fuel_weight_factor = 1

    airfoil_thicknesses = [
        xsec.airfoil.max_thickness()
        for xsec in wing.xsecs
    ]

    airfoil_t_over_c = np.min(airfoil_thicknesses)

    cos_sweep = np.cosd(wing.mean_sweep_angle())

    return (
            0.036 *
            (wing.area('planform') / u.foot ** 2) ** 0.758 *
            fuel_weight_factor *
            (wing.aspect_ratio() / cos_sweep ** 2) ** 0.6 *
            (cruise_op_point.dynamic_pressure() / u.psf) ** 0.006 *
            wing.taper_ratio() ** 0.04 *
            (100 * airfoil_t_over_c / cos_sweep) ** -0.3 *
            (design_mass_TOGW / u.lbm * ultimate_load_factor) ** 0.49 *
            (advanced_composites["wing"] if use_advanced_composites else 1)
    ) * u.lbm


def mass_hstab(
        hstab: asb.Wing,
        design_mass_TOGW: float,
        ultimate_load_factor: float,
        cruise_op_point: asb.OperatingPoint,
        use_advanced_composites: bool = False,
) -> float:
    """
    Computes the mass of a horizontal stabilizer of a general aviation aircraft, according to Raymer's Aircraft Design:
    A Conceptual Approach.

    Args:
        hstab: The horizontal stabilizer object.

        design_mass_TOGW: The design takeoff gross weight of the entire aircraft [kg].

        ultimate_load_factor: The ultimate load factor of the aircraft.

        cruise_op_point: The cruise operating point of the aircraft.

        use_advanced_composites: Whether to use advanced composites for the horizontal stabilizer. If True, the
        hstab mass is modified accordingly.

    Returns: The mass of the horizontal stabilizer [kg].
    """
    airfoil_thicknesses = [
        xsec.airfoil.max_thickness()
        for xsec in hstab.xsecs
    ]

    airfoil_t_over_c = np.min(airfoil_thicknesses)

    cos_sweep = np.cosd(hstab.mean_sweep_angle())

    return (
            0.016 *
            (design_mass_TOGW / u.lbm * ultimate_load_factor) ** 0.414 *
            (cruise_op_point.dynamic_pressure() / u.psf) ** 0.168 *
            (hstab.area('planform') / u.foot ** 2) ** 0.896 *
            (100 * airfoil_t_over_c / cos_sweep) ** -0.12 *
            (hstab.aspect_ratio() / cos_sweep ** 2) ** 0.043 *
            hstab.taper_ratio() ** -0.02 *
            (advanced_composites["tails"] if use_advanced_composites else 1)
    ) * u.lbm


def mass_vstab(
        vstab: asb.Wing,
        design_mass_TOGW: float,
        ultimate_load_factor: float,
        cruise_op_point: asb.OperatingPoint,
        is_t_tail: bool = False,
        use_advanced_composites: bool = False,
) -> float:
    """
    Computes the mass of a vertical stabilizer of a general aviation aircraft, according to Raymer's Aircraft Design:
    A Conceptual Approach.

    Args:
        vstab: The vertical stabilizer object.

        design_mass_TOGW: The design takeoff gross weight of the entire aircraft [kg].

        ultimate_load_factor: The ultimate load factor of the aircraft.

        cruise_op_point: The cruise operating point of the aircraft.

        is_t_tail: Whether the aircraft is a T-tail or not.

        use_advanced_composites: Whether to use advanced composites for the vertical stabilizer. If True, the vstab
        mass is modified accordingly.

    Returns: The mass of the vertical stabilizer [kg].
    """
    airfoil_thicknesses = [
        xsec.airfoil.max_thickness()
        for xsec in vstab.xsecs
    ]

    airfoil_t_over_c = np.min(airfoil_thicknesses)

    cos_sweep = np.cosd(vstab.mean_sweep_angle())

    return (
            0.073 *
            (1 + (0.2 if is_t_tail else 0)) *
            (design_mass_TOGW / u.lbm * ultimate_load_factor) ** 0.376 *
            (cruise_op_point.dynamic_pressure() / u.psf) ** 0.122 *
            (vstab.area('planform') / u.foot ** 2) ** 0.876 *
            (100 * airfoil_t_over_c / cos_sweep) ** -0.49 *
            (vstab.aspect_ratio() / cos_sweep ** 2) ** 0.357 *
            vstab.taper_ratio() ** 0.039 *
            (advanced_composites["tails"] if use_advanced_composites else 1)
    ) * u.lbm


def mass_fuselage(
        fuselage: asb.Fuselage,
        design_mass_TOGW: float,
        ultimate_load_factor: float,
        L_over_D: float,
        cruise_op_point: asb.OperatingPoint,
        wing_to_tail_distance: float,
        pressure_differential: float = 0.0,
        use_advanced_composites: bool = False,
) -> float:
    """
    Computes the mass of a fuselage of a general aviation aircraft, according to Raymer's Aircraft Design: A Conceptual
    Approach.

    Args:
        fuselage: The fuselage object.

        design_mass_TOGW: The design takeoff gross weight of the entire aircraft [kg].

        ultimate_load_factor: The ultimate load factor of the aircraft.

        L_over_D: The lift-to-drag ratio of the aircraft in cruise.

        cruise_op_point: The cruise operating point of the aircraft.

        wing_to_tail_distance: The distance between the wing root-quarter-chord-point and the tail
        root-quarter-chord-point of the aircraft [m].

        pressure_differential: The absolute value of the pressure differential across the fuselage [Pa].

        use_advanced_composites: Whether to use advanced composites for the fuselage. If True, the fuselage mass is
        modified accordingly.

    Returns: The mass of the fuselage [kg].
    """

    mass_fuselage_without_pressurization = (
                                                   0.052 *
                                                   (fuselage.area_wetted() / u.foot ** 2) ** 1.086 *
                                                   (design_mass_TOGW / u.lbm * ultimate_load_factor) ** 0.177 *
                                                   (wing_to_tail_distance / u.foot) ** -0.051 *
                                                   (L_over_D) ** -0.072 *
                                                   (cruise_op_point.dynamic_pressure() / u.psf) ** 0.241 *
                                                   (advanced_composites["fuselage/nacelle"]
                                                    if use_advanced_composites else 1)
                                           ) * u.lbm

    mass_pressurization_components = (
                                             11.9 *
                                             (
                                                     fuselage.volume() / u.foot ** 3 *
                                                     pressure_differential / u.psi
                                             ) ** 0.271
                                     ) * u.lbm

    return (
            mass_fuselage_without_pressurization +
            mass_pressurization_components
    )


def mass_main_landing_gear(
        main_gear_length: float,
        design_mass_TOGW: float,
        n_gear: int = 2,
        is_retractable: bool = True,
        use_advanced_composites: bool = False,
) -> float:
    """
    Computes the mass of the main landing gear of a general aviation aircraft, according to Raymer's Aircraft Design:
    A Conceptual Approach.

    Args:
        main_gear_length: The length of the main landing gear [m].

        design_mass_TOGW: The design takeoff gross weight of the entire aircraft [kg].

        n_gear: The number of main landing gear.

        is_retractable: Whether the main landing gear is retractable or not.

        use_advanced_composites: Whether to use advanced composites for the main landing gear. If True, the main
        landing gear mass is modified accordingly.

    Returns: The mass of the main landing gear [kg].
    """

    ultimate_landing_load_factor = n_gear * 1.5

    return (
            0.095 *
            (ultimate_landing_load_factor * design_mass_TOGW / u.lbm) ** 0.768 *
            (main_gear_length / u.foot / 12) ** 0.409 *
            (advanced_composites["landing_gear"] if use_advanced_composites else 1) *
            (((5.7 - 1.4 / 2) / 5.7) if not is_retractable else 1)  # derived from Raymer Section 15.2 and 15.3.3 together.
    ) * u.lbm


def mass_nose_landing_gear(
        nose_gear_length: float,
        design_mass_TOGW: float,
        n_gear: int = 1,
        is_retractable: bool = True,
        use_advanced_composites: bool = False,
) -> float:
    """
    Computes the mass of the nose landing gear of a general aviation aircraft, according to Raymer's Aircraft Design:
    A Conceptual Approach.

    Args:
        nose_gear_length: The length of the nose landing gear [m].

        design_mass_TOGW: The design takeoff gross weight of the entire aircraft [kg].

        n_gear: The number of nose landing gear.

        is_retractable: Whether the nose landing gear is retractable or not.

        use_advanced_composites: Whether to use advanced composites for the nose landing gear. If True, the nose
        landing gear mass is modified accordingly.

    Returns: The mass of the nose landing gear [kg].
    """

    ultimate_landing_load_factor = n_gear * 1.5

    return (
            0.125 *
            (ultimate_landing_load_factor * design_mass_TOGW / u.lbm) ** 0.566 *
            (nose_gear_length / u.foot / 12) ** 0.845 *
            (advanced_composites["landing_gear"] if use_advanced_composites else 1) *
            (((5.7 - 1.4 / 2) / 5.7) if not is_retractable else 1)  # derived from Raymer Section 15.2 and 15.3.3 together.
    ) * u.lbm


def mass_engines_installed(
        n_engines: int,
        mass_per_engine: float,
) -> float:
    """
    Computes the mass of the engines installed on a general aviation aircraft, according to Raymer's Aircraft Design:
    A Conceptual Approach. Includes propellers and engine mounts.

    Args:
        n_engines: The number of engines installed on the aircraft.

        mass_per_engine: The mass of a single engine [kg].

    Returns: The mass of the engines installed on the aircraft [kg].
    """
    return (
            2.575 *
            (mass_per_engine / u.lbm) ** 0.922 *
            n_engines
    ) * u.lbm


def mass_fuel_system(
        fuel_volume: float,
        n_tanks: int,
        n_engines: int,
        fraction_in_integral_tanks: float = 0.5,
) -> float:
    """
    Computes the mass of the fuel system (e.g., tanks, pumps, but not the fuel itself) for a general aviation
    aircraft, according to Raymer's Aircraft Design: A Conceptual Approach.

    Args:
        fuel_volume: The volume of fuel in the aircraft [m^3].

        n_tanks: The number of fuel tanks in the aircraft.

        n_engines: The number of engines in the aircraft.

        fraction_in_integral_tanks: The fraction of the fuel volume that is in integral tanks, as opposed to
        protected tanks.

    Returns: The mass of the fuel system [kg].
    """
    return (
            2.49 *
            (fuel_volume / u.gallon) ** 0.726 *
            (1 + fraction_in_integral_tanks) ** -0.363 *
            n_tanks ** 0.242 *
            n_engines ** 0.157
    ) * u.lbm


def mass_flight_controls(
        airplane: asb.Airplane,
        design_mass_TOGW: float,
        ultimate_load_factor: float,
        fuselage: asb.Fuselage = None,
        main_wing: asb.Wing = None,
) -> float:
    """
    Computes the mass of the flight controls for a general aviation aircraft, according to Raymer's Aircraft Design:
    A Conceptual Approach.

    Args:
        airplane: The airplane for which to compute the flight controls mass.

        design_mass_TOGW: The design takeoff gross weight of the entire aircraft [kg].

        ultimate_load_factor: The ultimate load factor of the aircraft.

        fuselage: The fuselage to use for computing the flight controls mass. If fuselage is None, or if there are no
        fuselages in the airplane object, the flight controls mass will be computed without a fuselage.

        main_wing: The main wing to use for computing the flight controls mass. If main_wing is None, or if there are
        no wings in the airplane object, the flight controls mass will be computed without a main wing.

    Returns: The mass of the flight controls [kg].
    """

    ### Handle the fuselage argument and get the fuselage length factor
    if fuselage is None:
        if len(airplane.fuselages) == 0:
            pass
        elif len(airplane.fuselages) == 1:
            fuselage = airplane.fuselages[0]
        else:
            raise ValueError('More than one fuselage is present in the airplane. Please specify which fuselage to use '
                             'for computing flight control system mass.')

    if fuselage is not None:
        fuselage_length_factor = (fuselage.length() / u.foot) ** 1.536
    else:
        fuselage_length_factor = 1

    ### Handle the main wing argument and get the wing span factor
    if main_wing is None:
        if len(airplane.wings) == 0:
            pass
        elif len(airplane.wings) == 1:
            main_wing = airplane.wings[0]
        else:
            raise ValueError('More than one wing is present in the airplane. Please specify which wing is the main'
                             'wing using the `main_wing` argument.')

    if main_wing is not None:
        wing_span_factor = (main_wing.span() / u.foot) ** 0.371
    else:
        wing_span_factor = 1

    # ### Compute how many functions the control surfaces are performing (e.g., aileron, elevator, flap, rudder, etc.)
    # N_functions_performed_by_controls = 0
    # for wing in airplane.wings:
    #     N_functions_performed_by_controls += len(wing.get_control_surface_names())
    #
    # ### Compute the control surface area
    # control_surface_area = 0
    # for wing in airplane.wings:
    #     control_surface_area += wing.control_surface_area()

    return (
            0.053 *
            fuselage_length_factor *
            wing_span_factor *
            (design_mass_TOGW / u.lbm * ultimate_load_factor * 1e-4) ** 0.80
    ) * u.lbm


def mass_hydraulics(
        fuselage_width: float,
        cruise_op_point: asb.OperatingPoint,
) -> float:
    """
    Computes the mass of the hydraulics for a general aviation aircraft, according to Raymer's Aircraft Design:
    A Conceptual Approach.

    Args:
        fuselage_width: The width of the fuselage [m].

        cruise_op_point: The cruise operating point of the aircraft.

    Returns: The mass of the hydraulics [kg].
    """
    mach = cruise_op_point.mach()

    K_h = 0.16472092991402892 * mach ** 0.8327375101470056
    # This is a curve fit to a few points that Raymer gives in his book. The points are:
    # {
    #     0.1 : 0.013,
    #     0.25: 0.05,
    #     0.5 : 0.11,
    #     0.75: 0.12
    # }
    # where the first column is the Mach number and the second column is the K_h value.
    # These are described as:
    #
    # "0.05 for low subsonic with hydraulics for brakes and retracts only; 0.11 for medium subsonic with hydraulics
    # for flaps; 0.12 for high subsonic with hydraulic flight controls; 0.013 for light plane with hydraulic brakes
    # only (and use M=0.1)"

    return (
            K_h *
            (fuselage_width / u.foot) ** 0.8 *
            mach ** 0.5
    ) * u.lbm


def mass_avionics(
        mass_uninstalled_avionics: float,
) -> float:
    """
    Computes the mass of the avionics for a general aviation aircraft, according to Raymer's Aircraft Design: A
    Conceptual Approach.

    Args:
        mass_uninstalled_avionics: The mass of the avionics, before installation [kg].

    Returns: The mass of the avionics, as installed [kg].
    """
    return (
            2.117 *
            (mass_uninstalled_avionics / u.lbm) ** 0.933
    ) * u.lbm


def mass_electrical(
        fuel_system_mass: float,
        avionics_mass: float,
) -> float:
    """
    Computes the mass of the electrical system for a general aviation aircraft, according to Raymer's Aircraft Design:
    A Conceptual Approach.

    Args:
        fuel_system_mass: The mass of the fuel system [kg].

        avionics_mass: The mass of the avionics [kg].

    Returns: The mass of the electrical system [kg].
    """

    fuel_and_avionics_masses = fuel_system_mass + avionics_mass

    return (
        12.57 *
        (fuel_and_avionics_masses / u.lbm) ** 0.51
    ) * u.lbm


def mass_air_conditioning_and_anti_ice(
        design_mass_TOGW: float,
        n_crew: int,
        n_pax: int,
        mass_avionics: float,
        cruise_op_point: asb.OperatingPoint,
):
    """
    Computes the mass of the air conditioning and anti-ice system for a general aviation aircraft, according to
    Raymer's Aircraft Design: A Conceptual Approach.

    Args:
        design_mass_TOGW: The design takeoff gross weight of the entire airplane [kg].

        n_crew: The number of crew members.

        n_pax: The number of passengers.

        mass_avionics: The mass of the avionics [kg].

        cruise_op_point: The cruise operating point of the aircraft.

    Returns: The mass of the air conditioning and anti-ice system [kg].
    """
    mach = cruise_op_point.mach()

    return (
            0.265 *
            (design_mass_TOGW / u.lbm) ** 0.52 *
            (n_crew + n_pax) ** 0.68 *
            (mass_avionics / u.lbm) ** 0.17 *
            mach ** 0.08
    ) * u.lbm


def mass_furnishings(
        design_mass_TOGW: float,
):
    """
    Computes the mass of the furnishings for a general aviation aircraft, according to Raymer's Aircraft Design: A
    Conceptual Approach.

    Args:
        design_mass_TOGW: The design takeoff gross weight of the entire airplane [kg].

    Returns: The mass of the furnishings [kg].
    """
    return np.softmax(
        0.0582 * design_mass_TOGW - 65 * u.lbm,
        0,
        softness=10 * u.lbm,
    )
