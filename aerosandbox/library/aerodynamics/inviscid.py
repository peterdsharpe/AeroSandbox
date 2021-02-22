import aerosandbox.numpy as np


def induced_drag(
        lift,
        span,
        dynamic_pressure,
        oswalds_efficiency=1,
):
    """
    Computes the induced drag associated with a lifting planar wing.

    Args:
        lift: Lift force [Newtons]
        span: Wing span [meters]
        dynamic_pressure: Dynamic pressure [Pascals]
        oswalds_efficiency: Oswald's efficiency factor [-]

    Returns: Induced drag force [Newtons]

    """
    return lift ** 2 / (
            dynamic_pressure * np.pi * span ** 2 * oswalds_efficiency
    )


def oswalds_efficiency(
        taper_ratio: float,
        aspect_ratio: float,
        sweep: float = 0.,
        fuselage_diameter_to_span_ratio: float = 0.,
) -> float:
    """
    Computes the Oswald's efficiency factor for a planar, tapered, swept wing.

    Based on "Estimating the Oswald Factor from Basic Aircraft Geometrical Parameters"
    by M. Nita, D. Scholz; Hamburg Univ. of Applied Sciences, 2012.

    Implementation of Section 5 from the above paper.

    Only valid for backwards-swept wings; i.e. 0 <= sweep < 90.

    Args:
        taper_ratio: Taper ratio of the wing (tip_chord / root_chord) [-]
        aspect_ratio: Aspect ratio of the wing (b^2 / S) [-]
        sweep: Wing quarter-chord sweep angle [deg]

    Returns: Oswald's efficiency factor [-]

    """

    def f(l):  # f(lambda), given as Eq. 36 in the Nita and Scholz paper (see parent docstring).
        return (
                0.0524 * l ** 4
                - 0.15 * l ** 3
                + 0.1659 * l ** 2
                - 0.0706 * l
                + 0.0119
        )

    delta_lambda = -0.357 + 0.45 * np.exp(-0.0375 * sweep)
    # Eq. 37 in Nita & Scholz.
    # Note: there is a typo in the cited paper; the negative in the exponent was omitted.
    # A bit of thinking about this reveals that this omission must be erroneous.

    e_theo = 1 / (
            1 + f(taper_ratio - delta_lambda) * aspect_ratio
    )

    fuselage_wake_contraction_correction_factor = 1 - 2 * (fuselage_diameter_to_span_ratio) ** 2

    e = e_theo * fuselage_wake_contraction_correction_factor

    return e


def optimal_taper_ratio(
        sweep=0.,
) -> float:
    """
    Computes the optimal (minimum-induced-drag) taper ratio for a given quarter-chord sweep angle.

    Based on "Estimating the Oswald Factor from Basic Aircraft Geometrical Parameters"
    by M. Nita, D. Scholz; Hamburg Univ. of Applied Sciences, 2012.

    Only valid for backwards-swept wings; i.e. 0 <= sweep < 90.

    Args:
        sweep: Wing quarter-chord sweep angle [deg]

    Returns: Optimal taper ratio

    """
    return 0.45 * np.exp(-0.0375 * sweep)


def CL_over_Cl(
        aspect_ratio: float,
        mach: float = 0.,
        sweep: float = 0.
) -> float:
    """
    Returns the ratio of 3D lift coefficient (with compressibility) to 2D lift coefficient (incompressible).
    :param aspect_ratio: Aspect ratio
    :param mach: Mach number
    :param sweep: Sweep angle [deg]
    :return:
    """
    beta = np.where(
        1 - mach ** 2 >= 0,
        np.fmax(1 - mach ** 2, 0) ** 0.5,
        0
    )
    # return aspect_ratio / (aspect_ratio + 2) # Equivalent to equation in Drela's FVA in incompressible, 2*pi*alpha limit.
    # return aspect_ratio / (2 + cas.sqrt(4 + aspect_ratio ** 2))  # more theoretically sound at low aspect_ratio
    eta = 0.95
    return aspect_ratio / (
            2 + np.sqrt(
        4 + (aspect_ratio * beta / eta) ** 2 * (1 + (np.tand(sweep) / beta) ** 2)
    )
    )  # From Raymer, Sect. 12.4.1; citing DATCOM


def induced_drag_ratio_from_ground_effect(
        h_over_b  # type: float
):
    """
    Gives the ratio of actual induced drag to free-flight induced drag experienced by a wing in ground effect.
    Artificially smoothed below around h/b == 0.05 to retain differentiability and practicality.
    Source: W. F. Phillips, D. F. Hunsaker, "Lifting-Line Predictions for Induced Drag and Lift in Ground Effect".
        Using Equation 5 from the paper, which is modified from a model from Torenbeek:
            Torenbeek, E. "Ground Effects", 1982.
    :param h_over_b: (Height above ground) divided by (wingspan).
    :return: Ratio of induced drag in ground effect to induced drag out of ground effect [unitless]
    """
    h_over_b = np.softmax(
        h_over_b,
        0,
        hardness=1 / 0.03
    )
    return 1 - np.exp(
        -4.01 * (2 * h_over_b) ** 0.717
    )
