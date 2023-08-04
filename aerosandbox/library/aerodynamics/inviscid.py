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
        method="nita_scholz",
) -> float:
    """
    Computes the Oswald's efficiency factor for a planar, tapered, swept wing.

    Based on "Estimating the Oswald Factor from Basic Aircraft Geometrical Parameters"
    by M. Nita, D. Scholz; Hamburg Univ. of Applied Sciences, 2012.
    https://www.fzt.haw-hamburg.de/pers/Scholz/OPerA/OPerA_PUB_DLRK_12-09-10.pdf

    Implementation of Section 5 from the above paper.

    Only valid for backwards-swept wings; i.e. 0 <= sweep < 90.

    Args:
        taper_ratio: Taper ratio of the wing (tip_chord / root_chord) [-]
        aspect_ratio: Aspect ratio of the wing (b^2 / S) [-]
        sweep: Wing quarter-chord sweep angle [deg]

    Returns: Oswald's efficiency factor [-]

    """
    sweep = np.clip(sweep, 0, 90)  # TODO input proper analytic continuation

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

    ### Correction factors, with nomenclature from Nita & Scholz
    k_e_F = 1 - 2 * (fuselage_diameter_to_span_ratio) ** 2
    k_e_D0 = np.mean([
        0.873,  # jet transport
        0.864,  # business jet
        0.804,  # turboprop
        0.804,  # general aviation
    ])
    k_e_M = 1
    # Compressibility correction not added because it only becomes significant well after M_crit, after which wave
    # drag dominates. Nita & Scholz also do not provide a model that extrapolates sensibly beyond M=0.9 or so,
    # so value is limited.

    if method == "nita_scholz":
        e = e_theo * k_e_F * k_e_D0 * k_e_M
    elif method == "kroo":
        mach_correction_factor = 1
        Q = 1 / (e_theo * k_e_F)
        from aerosandbox.library.aerodynamics.viscous import Cf_flat_plate
        P = 0.38 * Cf_flat_plate(Re_L=1e6)

        e = mach_correction_factor / (
                Q + P * np.pi * aspect_ratio
        )

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
    sweep = np.clip(sweep, 0, 90)  # TODO input proper analytic continuation
    return 0.45 * np.exp(-0.0375 * sweep)


def CL_over_Cl(
        aspect_ratio: float,
        mach: float = 0.,
        sweep: float = 0.,
        Cl_is_compressible: bool = True
) -> float:
    """
    Returns the ratio of 3D lift coefficient (with compressibility) to the 2D lift coefficient.

    Specifically: CL_3D / CL_2D

    Args:

        aspect_ratio: The aspect ratio of the wing.

        mach: The freestream Mach number.

        sweep: The sweep of the wing, in degrees. To be most accurate, this should be the sweep at the locus of
        thickest points along the wing.

        Cl_is_compressible: This flag indicates whether the 2D airfoil data already has compressibility effects
        modeled.

            For example:

                * If this flag is True, this function returns: CL_3D / CL_2D, where CL_2D is the sectional lift
                coefficient based on the local profile at the freestream mach number.

                * If this flag is False, this function returns: CL_3D / CL_2D_at_mach_zero, where CL_2D_... is the
                sectional lift coefficient based on the local profile at mach zero.

            For most accurate results, set this flag to True, and then model profile characteristics separately.

    """
    prandtl_glauert_beta_squared_ideal = 1 - mach ** 2

    # beta_squared = 1 - mach ** 2
    beta_squared = np.softmax(
        prandtl_glauert_beta_squared_ideal,
        -prandtl_glauert_beta_squared_ideal,
        hardness=3.0
    )

    ### Alternate formulations
    # CL_ratio = aspect_ratio / (aspect_ratio + 2) # Equivalent to equation in Drela's FVA in incompressible, 2*pi*alpha limit.
    # CL_ratio = aspect_ratio / (2 + np.sqrt(4 + aspect_ratio ** 2))  # more theoretically sound at low aspect_ratio

    ### Formulation from Raymer, Sect. 12.4.1; citing DATCOM.
    # Comparison to experiment suggests this is the most accurate.
    # Symbolically simplified to remove the PG singularity.
    eta = 1.0
    CL_ratio = aspect_ratio / (
            2 + (
            4 + (aspect_ratio ** 2 * beta_squared / eta ** 2) + (np.tand(sweep) * aspect_ratio / eta) ** 2
    ) ** 0.5
    )

    if Cl_is_compressible:
        CL_ratio = CL_ratio * beta_squared ** 0.5

    return CL_ratio


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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    fig, ax = plt.subplots()
    machs = np.linspace(0, 2, 500)
    plt.plot(machs, CL_over_Cl(5, machs, 0))
    p.show_plot()
