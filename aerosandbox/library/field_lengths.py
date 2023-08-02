import aerosandbox as asb
import aerosandbox.numpy as np
import aerosandbox.tools.units as u
from typing import Dict


def field_length_analysis_torenbeek(
        design_mass_TOGW: float,
        thrust_at_liftoff: float,
        lift_over_drag_climb: float,
        CL_max: float,
        s_ref: float,
        n_engines: int,
        atmosphere: asb.Atmosphere = None,
        CD_zero_lift: float = 0.03,
        obstacle_height: float = 35 * u.foot,
        friction_coefficient: float = 0.02,
        V_obstacle_over_V_stall: float = 1.3,
        minimum_V_liftoff_over_V_stall: float = 1.2,
        V_approach_over_V_stall: float = 1.3,
        maximum_braking_deceleration_g: float = 0.37,
        inertia_time: float = 4.5,
        approach_angle_deg: float = 3,
) -> Dict[str, float]:
    """
    Performs a field length analysis on an aircraft, returning a dictionary of field length parameters.

    Citations:

        * "Torenbeek": Egbert Torenbeek, "Synthesis of Subsonic Airplane Design", 1976. (Generally, section 5.4.5:
        Takeoff)

    Args:

        design_mass_TOGW: The takeoff gross weight of the entire aircraft [kg].

        thrust_at_liftoff: The thrust of the aircraft at the moment of liftoff [N].

        lift_over_drag_climb: The lift-to-drag ratio of the aircraft during the climb phase of takeoff [dimensionless].

        CL_max: The maximum lift coefficient of the aircraft [dimensionless]. Assumes any lift-augmentation devices (
        e.g., slats, flaps) are deployed.

        s_ref: The reference area of the aircraft [m^2].

        atmosphere: The atmosphere object to use for the analysis. Defaults to sea level.

        n_engines: The number of engines on the aircraft. Used during balanced field length calculation,
        which involves a single-engine-failure assumption.

        CD_zero_lift: The zero-lift drag coefficient of the aircraft [dimensionless].

        obstacle_height: The height of the obstacle clearance [m].

            Note:

                * FAR 23 requires a 50 foot obstacle clearance height.

                * FAR 25 requires a 35 foot obstacle clearance height.

        friction_coefficient: The coefficient of friction between the wheels and the runway.

            * 0.02 is a good value for a dry concrete runway.

            * 0.045 is a good value for short grass.

        V_obstacle_over_V_stall: The ratio of the airspeed while flying over the obstacle to the stall airspeed.

        minimum_V_liftoff_over_V_stall: The minimum-allowable ratio of the liftoff airspeed to the stall airspeed.

        V_approach_over_V_stall: The ratio of the approach airspeed to the stall airspeed.

        maximum_braking_deceleration_g: The maximum deceleration of the aircraft during braking [G].

            This is used when calculating the "brake" portion of the "accelerate-brake" balanced field length, as well
            as the braking during normal landing.

            * Standard brakes are around 0.37 G on dry concrete.

            * Advanced brakes with optimum brake pressure control, lift dumpers, and nosewheel braking can be as high
            as 0.5 G on dry concrete.

        inertia_time: The time it takes for the pilot and aircraft to collectively react to an engine failure during
        takeoff [seconds]. This is collectively the sum of:

            * The pilot's reaction time

            * The time it takes the other engines to spool down, in the event of a rejected takeoff and deceleration
            on the ground.

    Returns:
        A dictionary of field length parameters, including:

            * "takeoff_ground_roll_distance": The distance the aircraft will roll on the ground during a normal
            takeoff before liftoff [meters].

            * "takeoff_airborne_distance": The distance the aircraft will travel in the air during a normal takeoff [
            meters]. This is after liftoff, but before the aircraft has reached the obstacle clearance height.

            * "takeoff_total_distance": The total field length required during a normal takeoff [meters]. This
            includes both the ground roll itself, as well as the airborne distance before the obstacle clearance
            height is reached.

            * "balanced_field_length": The field length required for takeoff and obstacle clearance when one engine
            fails at "decision speed" [meters]. Decision speed is the speed during the ground roll at which,
            if an engine fails, the aircraft can either continue the takeoff or brake to a complete stop in the same
            total distance.

            * "landing_airborne_distance": The distance the aircraft will travel in the air during a normal landing
            before touchdown [meters]. Note that a normal landing involves passing the runway threshold at the
            specified obstacle clearance height.

            * "landing_ground_roll_distance": The distance the aircraft will roll on the ground after touchdown
            during a normal landing [meters].

            * "landing_total_distance": The total field length required during a normal landing [meters]. This
            includes both the airborne distance beyond the threshold that is required for obstacle clearance,
            as well as the ground roll distance after touchdown.

            * "V_stall": The stall speed of the aircraft at its takeoff gross weight [m/s].

            * "V_liftoff": The airspeed at the moment of liftoff during a normal takeoff [m/s].

            * "V_obstacle": The airspeed when the aircraft reaches the obstacle clearance height during a normal
            takeoff [m/s].

            * "V_approach": The airspeed when the aircraft reaches the runway threshold during a normal landing.

            * "V_touchdown": The airspeed when the aircraft touches down during a normal landing.

            * "flight_path_angle_climb": The flight path angle during a normal takeoff at the point when the airplane
            reaches the obstacle clearance height [radians].

            * "flight_path_angle_climb_one_engine_out": The flight path angle during a critical-engine-out takeoff at
            the point when the airplane reaches the obstacle clearance height [radians]. If this number is negative,
            engine failure results in inability to climb.

    """
    ### Set defaults
    if atmosphere is None:
        atmosphere = asb.Atmosphere(altitude=0)

    ### Constants
    g = 9.81  # m/s^2, gravitational acceleration

    ##### Normal takeoff analysis #####

    ### Compute TWR and climb physics
    thrust_over_weight_takeoff = thrust_at_liftoff / (design_mass_TOGW * g)

    flight_path_angle_climb = (
            thrust_over_weight_takeoff
            - 1 / lift_over_drag_climb
    )

    ### V_stall is the stall speed of the airplane.
    V_stall = np.sqrt(
        2 * design_mass_TOGW * g / (atmosphere.density() * s_ref * CL_max)
    )

    ### V_obstacle is the airspeed at the obstacle clearance height
    V_obstacle = V_obstacle_over_V_stall * V_stall

    ### V_liftoff is the airspeed at the moment of liftoff
    V_liftoff = V_obstacle * np.softmax(
        (1 + flight_path_angle_climb * 2 ** 0.5) ** -0.5,  # From Torenbeek
        minimum_V_liftoff_over_V_stall / V_obstacle_over_V_stall,
        hardness=1 / 0.01
    )

    takeoff_effective_friction_coefficient = (  # From Torenbeek, Eq. 5-76; an approximation
            friction_coefficient +
            0.72 * (CD_zero_lift / CL_max)
    )

    # From Torenbeek, Eq. 5-74
    takeoff_acceleration_g = thrust_over_weight_takeoff - takeoff_effective_friction_coefficient

    takeoff_ground_roll_distance = V_liftoff ** 2 / (
            2 * g * takeoff_acceleration_g
    )

    ### Compute the airborne distance required to clear the obstacle
    # From Torenbeek. Assumes an air maneuver after liftoff with CL=CL_liftoff and constant (thrust - drag).
    takeoff_airborne_distance = (
            (
                    V_liftoff ** 2 / (g * 2 ** 0.5)
            ) + (
                    obstacle_height / flight_path_angle_climb
            )
    )

    ### Compute the total distance required for normal takeoff, including obstacle clearance
    takeoff_total_distance = takeoff_ground_roll_distance + takeoff_airborne_distance

    ##### Balanced field length analysis #####

    if n_engines == 1:
        # If there is only one engine, the worst time *during the ground roll* for the engine to fail is right at liftoff.
        balanced_field_length = takeoff_ground_roll_distance + (
                V_liftoff ** 2 / (2 * g * maximum_braking_deceleration_g)
        )

        flight_path_angle_climb_one_engine_out = -1 / lift_over_drag_climb

    else:

        ### The flight path angle during a climb with one engine inoperative.
        flight_path_angle_climb_one_engine_out = (
                thrust_over_weight_takeoff * (n_engines - 1) / n_engines
                - 1 / lift_over_drag_climb
        )

        if n_engines == 2:
            minimum_allowable_flight_path_angle = 0.024
        elif n_engines == 3:
            minimum_allowable_flight_path_angle = 0.027
        elif n_engines >= 4:
            minimum_allowable_flight_path_angle = 0.030
        else:
            raise ValueError("`n_engines` must be an integer >= 1")

        # This is an approximation made by Torenbeek (Eq. 5-90, see citation in docstring)
        gamma_bar_takeoff = 0.06 + (flight_path_angle_climb_one_engine_out - minimum_allowable_flight_path_angle)

        air_density_ratio = atmosphere.density() / asb.Atmosphere(altitude=0).density()

        balanced_field_length = (  # From Torenbeek, Eq. 5-89, modified to have inertia distance scale with V_liftoff
                (V_liftoff ** 2 / (2 * g * (1 + gamma_bar_takeoff / maximum_braking_deceleration_g))) *
                (1 / takeoff_acceleration_g + 1 / maximum_braking_deceleration_g) *
                (1 + (2 * g * obstacle_height) / V_liftoff ** 2) +
                inertia_time * V_liftoff
        )

    # Do a softmax to make sure that the BFL is never shorter than the normal takeoff distance.
    balanced_field_length = np.softmax(
        balanced_field_length,
        takeoff_total_distance,
        softness=takeoff_total_distance / 100,
    )

    ##### Landing analysis #####

    # The factor of 2 is an approximation factor from Torenbeek, Section 5.4.6
    gamma_bar_landing = 2 * np.tand(approach_angle_deg)

    ### Compute the landing distance
    V_approach = V_approach_over_V_stall * V_stall
    V_touchdown = V_liftoff

    landing_airborne_distance = (  # From Torenbeek
                                        (V_approach ** 2 - V_touchdown ** 2) / (2 * g) + obstacle_height
                                ) / gamma_bar_landing

    landing_ground_roll_distance = (
            inertia_time * V_touchdown +
            V_touchdown ** 2 / (2 * g * maximum_braking_deceleration_g)
    )

    landing_total_distance = (
            landing_airborne_distance +
            landing_ground_roll_distance
    )

    return {
        "takeoff_ground_roll_distance"          : takeoff_ground_roll_distance,
        "takeoff_airborne_distance"             : takeoff_airborne_distance,
        "takeoff_total_distance"                : takeoff_total_distance,
        "balanced_field_length"                 : balanced_field_length,
        "landing_airborne_distance"             : landing_airborne_distance,
        "landing_ground_roll_distance"          : landing_ground_roll_distance,
        "landing_total_distance"                : landing_total_distance,
        "V_stall"                               : V_stall,
        "V_liftoff"                             : V_liftoff,
        "V_obstacle"                            : V_obstacle,
        "V_approach"                            : V_approach,
        "V_touchdown"                           : V_touchdown,
        "flight_path_angle_climb"               : flight_path_angle_climb,
        "flight_path_angle_climb_one_engine_out": flight_path_angle_climb_one_engine_out,
    }


def field_length_analysis(
        design_mass_TOGW: float,
        thrust_at_liftoff: float,
        lift_over_drag_climb: float,
        CL_max: float,
        s_ref: float,
        n_engines: int,
        V_engine_failure_balanced_field_length: float,
        atmosphere: asb.Atmosphere = None,
        CD_zero_lift: float = 0.03,
        obstacle_height: float = 35 * u.foot,
        friction_coefficient: float = 0.02,
        minimum_V_liftoff_over_V_stall: float = 1.2,
        maximum_braking_deceleration_g: float = 0.37,
        inertia_time: float = 2,
        approach_angle_deg: float = 3,
) -> Dict[str, float]:
    ### Set defaults
    if atmosphere is None:
        atmosphere = asb.Atmosphere(altitude=0)

    ### Constants
    g = 9.81  # m/s^2, gravitational acceleration

    ### Compute TWR and climb physics
    thrust_over_weight_takeoff = thrust_at_liftoff / (design_mass_TOGW * g)

    ##### Compute various accelerations
    acceleration_friction_and_drag = -g * (  # Based on Torenbeek, Eq. 5-76; an approximation
            friction_coefficient +
            0.72 * (CD_zero_lift / CL_max)
    )
    acceleration_braking = -g * maximum_braking_deceleration_g
    acceleration_engines = thrust_at_liftoff / design_mass_TOGW

    acceleration_takeoff = acceleration_engines + acceleration_friction_and_drag
    acceleration_coasting = acceleration_friction_and_drag
    acceleration_landing = acceleration_braking

    ##### Normal takeoff analysis #####

    ### V_stall is the stall speed of the airplane.
    V_stall = np.sqrt(
        2 * design_mass_TOGW * g / (atmosphere.density() * s_ref * CL_max)
    )

    ### V_liftoff is the airspeed at the moment of liftoff
    V_liftoff = minimum_V_liftoff_over_V_stall * V_stall

    takeoff_ground_roll_distance = V_liftoff ** 2 / (2 * acceleration_takeoff)

    ### Compute the airborne distance required to clear the obstacle
    flight_path_angle_climb = (  # radians, small angle approximation
            thrust_over_weight_takeoff
            - 1 / lift_over_drag_climb
    )
    flight_path_angle_climb = np.softmax(flight_path_angle_climb, 0, softness=0.001)

    takeoff_airborne_distance = obstacle_height / flight_path_angle_climb

    ### Compute the total distance required for normal takeoff, including obstacle clearance
    takeoff_total_distance = takeoff_ground_roll_distance + takeoff_airborne_distance

    ##### Normal landing analysis #####
    landing_airborne_distance = obstacle_height / np.tand(approach_angle_deg)

    V_touchdown = V_liftoff

    landing_ground_roll_distance = (
            inertia_time * V_touchdown +
            V_touchdown ** 2 / (2 * -acceleration_landing)
    )

    landing_total_distance = landing_airborne_distance + landing_ground_roll_distance

    ##### Balanced field length analysis #####

    if n_engines == 1:
        # If there is only one engine, the worst time *during the ground roll* for the engine to fail is right at liftoff.
        balanced_field_length = takeoff_ground_roll_distance + (
                V_liftoff ** 2 / (2 * -acceleration_landing)
        )

        balanced_field_length_accept = balanced_field_length
        balanced_field_length_reject = balanced_field_length

        flight_path_angle_climb_one_engine_out = -1 / lift_over_drag_climb

    else:
        acceleration_takeoff_one_engine_out = acceleration_engines * (
                n_engines - 1) / n_engines + acceleration_friction_and_drag

        ### The flight path angle during a climb with one engine inoperative.
        flight_path_angle_climb_one_engine_out = (
                thrust_over_weight_takeoff * (n_engines - 1) / n_engines
                - 1 / lift_over_drag_climb
        )
        flight_path_angle_climb_one_engine_out = np.softmax(flight_path_angle_climb_one_engine_out, 0, softness=0.001)

        balanced_field_length_accept = (
                (V_engine_failure_balanced_field_length ** 2 / (2 * acceleration_takeoff)) +  # Both engines working
                ((V_liftoff ** 2 - V_engine_failure_balanced_field_length ** 2) / (
                        2 * acceleration_takeoff_one_engine_out)) +
                (obstacle_height / flight_path_angle_climb_one_engine_out)
        )

        balanced_field_length_reject = (
                (V_engine_failure_balanced_field_length ** 2 / (2 * acceleration_takeoff)) +  # Both engines working
                (inertia_time * V_engine_failure_balanced_field_length) +  # Reaction time for pilot / engines
                (V_engine_failure_balanced_field_length ** 2 / (2 * -acceleration_landing))  # Braking time
        )

    return {
        "takeoff_ground_roll_distance"          : takeoff_ground_roll_distance,
        "takeoff_airborne_distance"             : takeoff_airborne_distance,
        "takeoff_total_distance"                : takeoff_total_distance,
        "balanced_field_length_accept"          : balanced_field_length_accept,
        "balanced_field_length_reject"          : balanced_field_length_reject,
        "landing_airborne_distance"             : landing_airborne_distance,
        "landing_ground_roll_distance"          : landing_ground_roll_distance,
        "landing_total_distance"                : landing_total_distance,
        "V_stall"                               : V_stall,
        "V_liftoff"                             : V_liftoff,
        "V_touchdown"                           : V_touchdown,
        "flight_path_angle_climb"               : flight_path_angle_climb,
        "flight_path_angle_climb_one_engine_out": flight_path_angle_climb_one_engine_out,
    }


if __name__ == '__main__':
    from aerosandbox.tools import units as u

    results = field_length_analysis(
        design_mass_TOGW=19000 * u.lbm,
        thrust_at_liftoff=19000 * u.lbf * 0.3,
        lift_over_drag_climb=20,
        CL_max=1.9,
        s_ref=24,
        n_engines=2,
        V_engine_failure_balanced_field_length=70,
        atmosphere=asb.Atmosphere(altitude=0),
    )

    results_torenbeek= field_length_analysis_torenbeek(
        design_mass_TOGW=19000 * u.lbm,
        thrust_at_liftoff=19000 * u.lbf * 0.3,
        lift_over_drag_climb=20,
        CL_max=1.9,
        s_ref=24,
        n_engines=2,
        atmosphere=asb.Atmosphere(altitude=0),
    )

    from pprint import pprint

    pprint(results)
