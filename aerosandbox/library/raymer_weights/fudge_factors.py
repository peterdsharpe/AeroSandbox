# From Raymer, Aircraft Design: A Conceptual Approach, 5th Ed.
# Table 15.4
advanced_composites = {
    # Format:
    # "component"         : (low, high)
    "wing"                : (0.85, 0.90),
    "tails"               : (0.83, 0.88),
    "fuselage/nacelle"    : (0.90, 0.95),
    "landing_gear"        : (0.95, 1),
    "air_induction_system": (0.85, 0.90),
}

advanced_composites = { # Here, we convert this to a dictionary of average values.
    k: (v[0] + v[1]) / 2
    for k, v in advanced_composites.items()
}

braced_wing = 0.82

braced_biplane = 0.6

wood_fuselage = 1.60

steel_tube_fuselage = 1.80

flying_boat_hull = 1.25

carrier_based_fuselage = 1.25
carrier_based_landing_gear = 1.25
