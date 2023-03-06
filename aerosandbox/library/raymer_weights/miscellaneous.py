import aerosandbox as asb
import aerosandbox.numpy as np
import aerosandbox.tools.units as u

# From Raymer, Aircraft Design: A Conceptual Approach, 5th Ed.
# Table 15.3: Miscellaneous Weights

mass_passenger = 215 * u.lbm  # includes carry-on


def mass_seat(
        kind="passenger"
) -> float:
    """
    Computes the mass of an individual seat on an airplane.

    Args:

        kind: The kind of seat. Can be "passenger", "flight_deck", or "troop".

            * "passenger" seats are standard commercial airline seats.

            * "flight_deck" seats are the seats in the cockpit.

            * "troop" seats are the seats in the cargo hold.

    Returns: The mass of a single seat, in kg. Don't forget to multiply by the number of seats to get the total mass
    of all seats.
    """
    if kind == "passenger":
        return 32 * u.lbm
    elif kind == "flight_deck":
        return 60 * u.lbm
    elif kind == "troop":
        return 11 * u.lbm
    else:
        raise ValueError("Bad value of `kind`!")


def mass_lavatories(
        n_pax,
        aircraft_type="short-haul"
) -> float:
    """
    Computes the required mass of all lavatories on an airplane.

    Args:
        n_pax: The number of passengers on the airplane.

        aircraft_type: The type of aircraft. Can be "long-haul", "short-haul", or "business-jet".

            * "long-haul" aircraft are long-range commercial airliners, like the Boeing 777 or Airbus A350.

            * "short-haul" aircraft are short-range commercial airliners, like the Boeing 737 or Airbus A320.

            * "business-jet" aircraft are small private jets, like the Cessna Citation X or Gulfstream G650.

    Returns: The mass of all lavatories on the airplane, in kg.

    """
    if aircraft_type == "long-haul":
        return (1.11 * n_pax ** 1.33) * u.lbm
    elif aircraft_type == "short-haul":
        return (0.31 * n_pax ** 1.33) * u.lbm
    elif aircraft_type == "business-jet":
        return (3.90 * n_pax ** 1.33) * u.lbm
    else:
        raise ValueError("Bad value of `aircraft_type`!")

