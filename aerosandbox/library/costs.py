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

        n_airplanes_produced: The number of airplanes produced or the number to be produced in 5 years, whichever is
        less.

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
        A dictionary of costs, to produce all `n_airplanes_produced` airplanes, in present-day dollars.

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

            - "engine_cost": Engine cost.

            - "avionics_cost": Avionics cost.

            - "total_cost": Total cost.

    """
    # Abbreviated constants for readability
    W = design_empty_weight  # kg
    V = design_maximum_airspeed / u.kph  # km/hour
    Q = n_airplanes_produced

    ### Estimate labor hours
    hours = {}

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
    costs_2012_dollars = {}

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

    ### Add engine and avionics costs
    costs["engine_cost"] = cost_per_engine * n_engines_per_aircraft * n_airplanes_produced
    costs["avionics"] = cost_avionics_per_airplane * n_airplanes_produced

    ### Total all costs and return
    costs["total"] = sum(costs.values())

    return costs
