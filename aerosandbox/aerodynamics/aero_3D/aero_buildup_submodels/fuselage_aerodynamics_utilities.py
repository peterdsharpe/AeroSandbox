
def critical_mach(fineness_ratio_nose: float) -> float:
    """
    Returns the transonic critical Mach number for a streamlined fuselage.

    Fitted to data from Raymer "Aircraft Design: A Conceptual Approach" 2nd Ed., Fig. 12.28.
    See figure + study + fit in: /studies/FuselageCriticalMach/

    Args:

        fineness_ratio_nose: The fineness ratio of the nose section of the fuselage.

            Specifically, fineness_ratio_nose = 2 * L_n / d, where:

                * L_n is the length from the nose to the longitudinal location at which the fuselage cross section
                becomes essentially constant, and:

                * d is the body diameter at that location.

    Returns: The critical Mach number

    """
    p = {
        'a': 11.087202397070559,
        'b': 13.469755774708842,
        'c': 4.034476257077558
    }

    mach_dd = 1 - (p["a"] / (2 * fineness_ratio_nose + p["b"])) ** p["c"]

    ### The following approximate relation is derived in W.H. Mason, "Configuration Aerodynamics", Chapter 7. Transonic Aerodynamics of Airfoils and Wings.
    ### Equation 7-8 on Page 7-19.
    ### This is in turn based on Lock's proposed empirically-derived shape of the drag rise, from Hilton, W.F., High Speed Aerodynamics, Longmans, Green & Co., London, 1952, pp. 47-49
    mach_crit = mach_dd - (0.1 / 80) ** (1 / 3)

    return mach_crit


def jorgensen_eta(fineness_ratio: float) -> float:
    """
    A fit for the eta parameter (crossflow lift multiplier) of a fuselage, as described in:

    Jorgensen, Leland Howard. "Prediction of Static Aerodynamic Characteristics for Slender Bodies
    Alone and with Lifting Surfaces to Very High Angles of Attack". NASA TR R-474. 1977.

    Fits performed in /studies/JorgensenEtaFitting/

    Args:
        fineness_ratio: The fineness ratio of the fuselage. (length / diameter)

    Returns: An estimate of eta.

    """
    x = fineness_ratio
    p = {
        '1scl': 23.009059965179222,
        '1cen': -122.76900250914575,
        '2scl': 13.006453125841258,
        '2cen': -24.367562906887436
    }
    return 1 - p["1scl"] / (x - p["1cen"]) - (p["2scl"] / (x - p["2cen"])) ** 2