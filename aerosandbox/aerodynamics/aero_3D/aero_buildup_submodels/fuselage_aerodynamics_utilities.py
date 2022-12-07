import aerosandbox.numpy as np


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

    Fits performed in /studies/FuselageJorgensenEtaFitting/

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


def fuselage_base_drag_coefficient(mach: float) -> float:
    """
    A fit for the fuselage base drag coefficient of a cylindrical fuselage, as described in:

    MIL-HDBK-762: DESIGN OF AERODYNAMICALLY STABILIZED FREE ROCKETS:
        * Section 5-5.3.1 Body-of-Revolution Base Drag, Rocket Jet Plume-Off
        * Figure 5-140: Effects of Mach Number and Reynolds Number on Base Pressure

    Fits in /studies/FuselageBaseDragCoefficient

    Args:
        mach: Mach number [-]

    Returns: Fuselage base drag coefficient

    """

    m = mach
    p = {'a'         : 0.18024110740341143,
         'center_sup': -0.21737019935624047,
         'm_trans'   : 0.9985447737532848,
         'pc_sub'    : 0.15922582283573747,
         'pc_sup'    : 0.04698820458826384,
         'scale_sup' : 0.34978926411193456,
         'trans_str' : 9.999987483414937}

    return np.blend(
        p["trans_str"] * (m - p["m_trans"]),
        p["pc_sup"] + p["a"] * np.exp(-(p["scale_sup"] * (m - p["center_sup"])) ** 2),
        p["pc_sub"]
    )


def fuselage_form_factor(
        fineness_ratio: float,
        ratio_of_corner_radius_to_body_width: float = 0.5
):
    """
    Computes the form factor of a fuselage as a function of various geometrical parameters.

    Assumes the body cross section is a rounded square with constant-radius-of-curvature fillets.
    Body cross section can therefore vary from a true square to a true circle.

    Uses the methodology described in:

    Götten, Falk; Havermann, Marc; Braun, Carsten; Marino, Matthew; Bil, Cees.
    "Improved Form Factor for Drag Estimation of Fuselages with Various Cross Sections.
    AIAA Journal of Aircraft, 2021. DOI: 10.2514/1.C036032

    https://arc.aiaa.org/doi/10.2514/1.C036032

    Assumes fully turbulent flow. Coefficient of determination found in the paper above was 0.95.

    Note: the value returned does not account for any base separation (other than minor aft-closure separation). The
    equations were also fit to relatively-shape-optimized fuselages, and will be overly-optimistic for unoptimized
    shapes.

    Args:

        fineness_ratio: The fineness ratio of the body (length / diameter).

        ratio_of_corner_radius_to_body_width: A parameter that describes the cross-sectional shape of the fuselage.
        Precisely, this is ratio of corner radius to body width.

            * A value of 0 corresponds to a true square.

            * A value of 0.5 (default) corresponds to a true circle.

    Returns: The form factor of the body, defined as:

        C_D = C_f * form_factor * (S_wet / S_ref)

    """
    fr = fineness_ratio
    r = 2 * ratio_of_corner_radius_to_body_width

    cs1 = -0.825885 * r ** 0.411795 + 4.0001
    cs2 = -0.340977 * r ** 7.54327 - 2.27920
    cs3 = -0.013846 * r ** 1.34253 + 1.11029

    form_factor = cs1 * fr ** cs2 + cs3

    return form_factor
