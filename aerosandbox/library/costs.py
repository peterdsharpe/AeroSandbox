import aerosandbox.numpy as np
import aerosandbox.tools.units as u
from typing import Dict


def modified_DAPCA_IV_production_cost_analysis(
        design_empty_weight: float,
        design_maximum_airspeed: float,
        n_airplanes_produced: int,
        n_engines_per_aircraft: int,
        cost_per_engine: float,
        cost_avionics_per_airplane: float,
        n_pax: int,
        cpi_relative_to_2012_dollars: float = 1.275,  # updated for 2022
        n_flight_test_aircraft: int = 4,
        is_cargo_airplane: bool = False,
        primary_structure_material: str = "aluminum",
        per_passenger_cost_model: str = "general_aviation",
        engineering_wrap_rate_2012_dollars: float = 115.,
        tooling_wrap_rate_2012_dollars: float = 118.,
        quality_control_wrap_rate_2012_dollars: float = 108.,
        manufacturing_wrap_rate_2012_dollars: float = 98.,
) -> Dict[str, float]:
    """
    Computes the cost of an aircraft in present-day dollars, using the Modified DAPCA IV cost model.

    Be sure to adjust `cpi_relative_to_2012_dollars` to the current values in order to accurately model inflation.

    The DAPCA IV cost model is a statistical regression of historical aircraft cost data. It provides reasonable
    results for most classes of aircraft, including transports, fighters, bombers, and even GA and UAV aircraft with
    suitable adjustments.

    It was created by the RAND Corporation.

    The Modified DAPCA IV cost model is a modification of the DAPCA IV cost model that includes additional cost
    estimates for engine cost (as the original DAPCA model assumes that this is known).

    See Raymer, Aircraft Design: A Conceptual Approach, 5th Edition, Section 18.4.2 pg. 711 for more information.

    Args:
        design_empty_weight: The design empty weight of the entire aircraft, in kg.

        design_maximum_airspeed: The design maximum airspeed of the aircraft, in m/s.

        n_airplanes_produced: The number of airplanes to be produced.

        n_engines_per_aircraft: The number of engines per aircraft.

        cost_per_engine: The cost of each engine, in present-day dollars.

        cost_avionics_per_airplane: The cost of avionics per airplane, in present-day dollars.

        n_pax: The number of passengers.

        cpi_relative_to_2012_dollars: The consumer price index at the present day divided by the consumer price index
        in 2012, seasonally-adjusted.

            To quickly find this, use data from the St. Louis Federal Reserve. Below is the CPI, normalized to 2012.
            https://fred.stlouisfed.org/graph/?g=10PU0

            For example, in 2022, one would use 1.275.

        n_flight_test_aircraft: The number of flight test aircraft. Typically 2 to 6.

        is_cargo_airplane: Whether the airplane is a cargo airplane. If so, the quality control cost is lower.

        primary_structure_material: The primary structure material. Options are:
            - "aluminum"
            - "carbon_fiber"
            - "fiberglass"
            - "steel"
            - "titanium"

        per_passenger_cost_model: The per-passenger cost model. Options are:
            - "general_aviation": General aviation aircraft, such as Cessna 172s.
            - "jet_transport": Jet transport aircraft, such as Boeing 737s.
            - "regional_transport": Regional transport aircraft, such as Embraer E175s.

        engineering_wrap_rate_2012_dollars: The engineering wrap rate in 2012 dollars.

        tooling_wrap_rate_2012_dollars: The tooling wrap rate in 2012 dollars.

        quality_control_wrap_rate_2012_dollars: The quality control wrap rate in 2012 dollars.

        manufacturing_wrap_rate_2012_dollars: The manufacturing wrap rate in 2012 dollars.

    Returns:

        A dictionary of costs required to produce all `n_airplanes_produced` airplanes, in present-day dollars.

        Keys and values are as follows:

            - "engineering_labor": Engineering labor cost.

            - "tooling_labor": Tooling labor cost.

            - "manufacturing_labor": Manufacturing labor cost.

            - "quality_control_labor": Quality control labor cost.

            - "development_support": Development support cost. From Raymer: "Includes fabrication of mockups, iron-bird subsystem
            simulators, structural test articles, and other test articles."

            - "flight_test": Flight test cost. From Raymer: "Includes all costs incurred to demonstrate airworthiness
            for civil certification or Mil-Spec compliance except for the costs of the flight-test aircraft
            themselves. Costs for the flight-test aircraft are included in the total production-run cost estimation.
            Includes planning, instrumentation, flight operations, data reduction, and engineering and manufacturing
            support of flight testing."

            - "manufacturing_materials": Manufacturing materials cost. From Raymer: "Includes all raw materials and
            purchased hardware and equipment."

            - "engines": Engine cost.

            - "avionics": Avionics cost.

            - "total": Total cost. (Sum of all other costs above.)

    """
    # Abbreviated constants for readability
    W = design_empty_weight  # kg
    V = design_maximum_airspeed / u.kph  # km/hour
    Q = n_airplanes_produced

    ### Estimate labor hours
    hours = dict()

    hours["engineering"] = 5.18 * W ** 0.777 * V ** 0.894 * Q ** 0.163
    hours["tooling"] = 7.22 * W ** 0.777 * V ** 0.696 * Q ** 0.263
    hours["manufacturing"] = 10.5 * W ** 0.82 * V ** 0.484 * Q ** 0.641
    hours["quality_control"] = hours["manufacturing"] * (0.076 if is_cargo_airplane else 0.133)

    ### Account for materials difficulties
    if primary_structure_material == "aluminum":
        materials_hourly_multiplier = 1.0
    elif primary_structure_material == "carbon_fiber":
        materials_hourly_multiplier = (1.1 + 1.8) / 2
    elif primary_structure_material == "fiberglass":
        materials_hourly_multiplier = (1.1 + 1.2) / 2
    elif primary_structure_material == "steel":
        materials_hourly_multiplier = (1.5 + 2.0) / 2
    elif primary_structure_material == "titanium":
        materials_hourly_multiplier = (1.1 + 1.8) / 2
    else:
        raise ValueError("Invalid value of `primary_structure_material`.")

    hours = {
        k: v * materials_hourly_multiplier
        for k, v in hours.items()
    }

    ### Convert labor hours to labor costs in 2012 dollars
    costs_2012_dollars = dict()

    costs_2012_dollars["engineering_labor"] = hours["engineering"] * engineering_wrap_rate_2012_dollars
    costs_2012_dollars["tooling_labor"] = hours["tooling"] * tooling_wrap_rate_2012_dollars
    costs_2012_dollars["manufacturing_labor"] = hours["manufacturing"] * manufacturing_wrap_rate_2012_dollars
    costs_2012_dollars["quality_control_labor"] = hours["quality_control"] * quality_control_wrap_rate_2012_dollars

    costs_2012_dollars["development_support"] = 67.4 * W ** 0.630 * V ** 1.3
    costs_2012_dollars["flight_test"] = 1947 * W ** 0.325 * V ** 0.822 * n_flight_test_aircraft ** 1.21
    costs_2012_dollars["manufacturing_materials"] = 31.2 * W ** 0.921 * V ** 0.621 * Q ** 0.799

    ### Add in the per-passenger cost for aircraft interiors:
    # Seats, luggage bins, closets, lavatories, insulation, ceilings, floors, walls, etc.
    # Costs are from Raymer, Aircraft Design: A Conceptual Approach, 5th edition. Section 18.4.2, page 715.
    if per_passenger_cost_model == "general_aviation":
        costs_2012_dollars["aircraft_interiors"] = n_airplanes_produced * n_pax * 850
    elif per_passenger_cost_model == "jet_transport":
        costs_2012_dollars["aircraft_interiors"] = n_airplanes_produced * n_pax * 3500
    elif per_passenger_cost_model == "regional_transport":
        costs_2012_dollars["aircraft_interiors"] = n_airplanes_produced * n_pax * 1700
    else:
        raise ValueError(f"Invalid value of `per_passenger_cost_model`!")

    ### Convert all costs to present-day dollars
    costs = {
        k: v * cpi_relative_to_2012_dollars
        for k, v in costs_2012_dollars.items()
    }

    ### Add the engine(s) and avionics costs
    costs["engines"] = cost_per_engine * n_engines_per_aircraft * n_airplanes_produced
    costs["avionics"] = cost_avionics_per_airplane * n_airplanes_produced

    ### Total all costs and return
    costs["total"] = sum(costs.values())

    return costs


def electric_aircraft_direct_operating_cost_analysis(
        production_cost_per_airframe: float,
        nominal_cruise_airspeed: float,
        nominal_mission_range: float,
        battery_capacity: float,
        num_passengers_nominal: int,
        num_crew: int = 1,
        battery_fraction_used_on_nominal_mission: float = 0.8,
        typical_passenger_utilization: float = 0.8,
        flight_hours_per_year: float = 1200,
        airframe_lifetime_years: float = 20,
        airframe_eol_resale_value_fraction: float = 0.4,
        battery_cost_per_kWh_capacity: float = 500.0,
        battery_cycle_life: float = 1500,
        real_interest_rate: float = 0.04,
        electricity_cost_per_kWh: float = 0.145,
        annual_expenses_per_crew: float = 100000 * 1.5,
        ascent_time: float = 0.2 * u.hour,
        descent_time: float = 0.2 * u.hour,
) -> Dict[str, float]:
    """
    Estimates the overall operating cost of an electric aircraft. Includes both direct and indirect operating costs.

    Here, direct operating costs (DOC) are taken to include the following costs:

    - Airframe depreciation
    - Airframe financing
    - Insurance
    - Maintenance
    - Battery replacement
    - Energy costs (here, electricity)
    - Cockpit and cabin crew costs
    - Airport landing, terminal, and handling fees

    Any costs that are not included here are considered indirect costs. These indirect costs would include,
    but are not limited to: advertisement, administrative costs, depreciation of non-airframe assets, and taxes.

    Airframe maintenance costs are estimated from:
        Moore, et al., "Unlocking Low-Cost Regional Air Mobility through...", AIAA Aviation 2023.

    Airport fees estimated for the Phoenix-Mesa Gateway Airport, due to public availability of the fee schedule:
    https://www.gatewayairport.com/documents/documentlibrary/wgaa%20organizational%20documents/airport%20rates%20charges%20-%20effective%20march%201,%202017.pdf

    Args:

        production_cost_per_airframe: The cost to produce a single airframe, in present-day dollars. May be estimated
            using the `modified_DAPCA_IV_production_cost_analysis()` function.

        nominal_cruise_airspeed: The nominal cruise airspeed of the aircraft, in m/s.

        nominal_mission_range: The nominal mission range of the aircraft, in meters.

        battery_capacity: The total capacity of the battery, in Joules.

        num_passengers_nominal: The number of passengers that the aircraft is designed to carry.

        num_crew: The number of crew members required to operate the aircraft.

        battery_fraction_used_on_nominal_mission: The fraction of the battery's capacity that is used on the nominal
            mission.

        typical_passenger_utilization: The fraction of the aircraft's passenger capacity that is typically utilized.

        flight_hours_per_year: The number of flight hours per year that the aircraft is expected to fly.

        airframe_lifetime_years: The number of years that the airframe is expected to last. After this time, the airframe
            is assumed to be sold at some lower reasle value.

        airframe_eol_resale_value_fraction: The expect resale value of the airframe at the end of its lifetime,
            expressed as a fraction of the airframe's production cost.

        battery_cost_per_kWh_capacity: The replacement cost of the battery pack, per kWh of capacity, in present-day
            dollars. Note that this is a pack-level cost (as opposed to cell-level), so includes the cost of the
            battery management system, cooling system, fire-suppressing foam, etc. Should include the labor cost to
            replace the battery pack as well.

        battery_cycle_life: The number of charge/discharge cycles that the battery is expected to last before full
            replacement is required.

        real_interest_rate: The real interest rate per year. This is the interest rate minus the inflation rate. This is
            used to calculate the present-day value of future costs (e.g., airframe financing).

        electricity_cost_per_kWh: The cost of electricity, per kWh, in present-day dollars.

        annual_expenses_per_crew: The annual expenses per crew member, in present-day dollars. Should include the total
            burdened cost of the crew member, including salary, benefits, and other expenses.

        ascent_time: The time required to ascend to cruise altitude, in seconds.

        descent_time: The time required to descend from cruise altitude, in seconds.

    Returns:
        A dictionary of operating costs per passenger-mile, in present-day dollars, with the following keys:

            * "airframe_depreciation"
            * "airframe_financing"
            * "insurance"
            * "airframe_maintenance"
            * "propulsion_maintenance"
            * "battery_replacement"
            * "energy"
            * "crew"
            * "airport_landing_fees"
            * "airport_terminal_fees"
            * "airport_parking_fees"
            * "airport_passenger_facility_charge"
            * "indirect"

        One key, "total", is also included, which is the sum of all of the above costs. Once again, this is
            expressed in units of present-day dollars per passenger-mile.

    """

    ### Calculate per-mission parameters
    ascent_and_descent_airspeed = 0.75 * nominal_cruise_airspeed
    ascent_and_descent_time = np.softmin(
        ascent_time + descent_time,
        nominal_mission_range / ascent_and_descent_airspeed,
        softness=2 * u.minute,
    )
    ascent_and_descent_distance = ascent_and_descent_airspeed * ascent_and_descent_time
    cruise_distance = nominal_mission_range - ascent_and_descent_distance
    cruise_time = cruise_distance / nominal_cruise_airspeed
    mission_time = ascent_and_descent_time + cruise_time

    ### Calculate annual utilization parameters
    flights_per_year = flight_hours_per_year / (mission_time / u.hour)

    ### Begin calculating costs on a per-flight basis
    costs_per_flight = dict()  # Lists the cost per flight for each cost category

    ### Airframe depreciation costs
    num_airframe_flights_lifetime = flights_per_year * airframe_lifetime_years
    net_value_per_airframe_over_lifetime = (
            production_cost_per_airframe  # Production cost
            - (production_cost_per_airframe * airframe_eol_resale_value_fraction)  # End-of-life resale value
    )
    costs_per_flight["airframe_depreciation"] = (
            net_value_per_airframe_over_lifetime / num_airframe_flights_lifetime
    )

    ### Airframe financing cost
    airframe_financing_lifetime_cost = production_cost_per_airframe * (
            np.exp(real_interest_rate * airframe_lifetime_years) - 1
    )
    costs_per_flight["airframe_financing"] = (
            airframe_financing_lifetime_cost / num_airframe_flights_lifetime
    )

    ### Insurance cost
    insurance_cost_per_year = production_cost_per_airframe * (
            0.025  # Base rate of 2% on average, adjusted higher for perceived higher risk of new electric technology
            * ((num_passengers_nominal + num_crew + 1) / 11) ** 0.5  # Adj. for number of souls on board; but sublinear
            * (flight_hours_per_year / 1200) ** 0.5  # Normalizing to industry average; but again, sublinear
    )
    costs_per_flight["insurance"] = insurance_cost_per_year / flights_per_year

    ### Airframe maintenance cost
    # maintenance_cost_per_year = production_cost_per_airframe * 0.04
    # costs_per_flight["maintenance"] = maintenance_cost_per_year / flights_per_year
    costs_per_flight["airframe_maintenance"] = (production_cost_per_airframe / 3e6) * (
            65 * (mission_time / u.hour) +
            (65)  # per cycle
    )

    ### Propulsion maintenance cost
    costs_per_flight["propulsion_maintenance"] = (production_cost_per_airframe / 3e6) * (
            (58 * (mission_time / u.hour)) +
            (50)  # per cycle
    )

    ### Battery replacement cost
    battery_capacity_kWh = battery_capacity / (u.kilo * u.watt_hour)
    battery_cost = battery_capacity_kWh * battery_cost_per_kWh_capacity
    costs_per_flight["battery_replacement"] = battery_cost / battery_cycle_life

    ### Energy Cost
    electric_energy_per_flight = battery_capacity * battery_fraction_used_on_nominal_mission
    electric_energy_per_flight_kWh = electric_energy_per_flight / (u.kilo * u.watt_hour)
    costs_per_flight["energy"] = electric_energy_per_flight_kWh * electricity_cost_per_kWh

    ### Crew cost
    costs_per_flight["crew"] = (
            num_crew *
            annual_expenses_per_crew /
            flights_per_year
    )

    ### Airport landing fees
    estimated_max_gross_landing_weight = (  # Model is very approximate, because landing fees are small
            production_cost_per_airframe / 266  # typical cost per kg empty weight
            * 1.2  # Rough ratio of landing weight to empty weight
    )
    costs_per_flight["airport_landing_fees"] = (
            1.20 * estimated_max_gross_landing_weight / 1000
    )

    ### Airport terminal use fees
    costs_per_flight["airport_terminal_fees"] = (
        50  # Cost per turn-around
    )

    ### Airport aircraft parking fees
    costs_per_flight["airport_parking_fees"] = (
        35  # Cost per turn-around
    )

    ### Airport passenger facility charge
    costs_per_flight["airport_passenger_facility_charge"] = (
            4.50 * num_passengers_nominal
    )

    ### Indirect Costs
    costs_per_flight["indirect"] = 0.2 * sum(costs_per_flight.values())

    # Cost per passenger mile
    n_passengers_actual = num_passengers_nominal * typical_passenger_utilization
    passenger_miles_per_flight = n_passengers_actual * (nominal_mission_range / u.mile)
    costs_per_flight["total"] = sum(costs_per_flight.values())

    # Return same dictionary as before
    costs_per_paxmi = {
        k: v / passenger_miles_per_flight
        for k, v in costs_per_flight.items()
    }

    return costs_per_paxmi

if __name__ == '__main__':
    res = electric_aircraft_direct_operating_cost_analysis(
        production_cost_per_airframe=3.0e6,
        nominal_cruise_airspeed=250 * u.knot,
        nominal_mission_range=150 * u.mile,
        battery_capacity=800e3 * u.watt_hour,
        num_passengers_nominal=9,
    )
    import pandas as pd

    print(pd.Series(res))
