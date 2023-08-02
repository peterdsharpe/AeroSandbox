import aerosandbox.numpy as np
from aerosandbox.modeling.splines.hermite import linear_hermite_patch, cubic_hermite_patch


def sears_haack_drag(
        radius_max: float,
        length: float
) -> float:
    """
    Yields the idealized drag area (denoted CDA, or equivalently, D/q) of a Sears-Haack body.

    Assumes linearized supersonic (Prandtl-Glauert) flow.

    https://en.wikipedia.org/wiki/Sears%E2%80%93Haack_body

    Note that drag coefficient and drag area are independent of Mach number for this case (assuming linearized supersonic aero).

    Args:
        radius_max: The maximum radius of the Sears-Haack body.
        length: The length of the Sears-Haack body.

    Returns: The drag area (CDA, or D/q) of the body. To get the drag force, multiply by the dynamic pressure.

    """
    CDA = 9 * np.pi ** 2 * radius_max ** 2 / (2 * length ** 2)
    return CDA


def sears_haack_drag_from_volume(
        volume: float,
        length: float
) -> float:
    """
    See documentation for sears_haack_drag() in this same file.

    Identical, except takes volume as an input rather than max radius.

    Also returns a drag area (denoted CDA, or equivalently, D/q).
    """
    CDA = 128 * volume ** 2 / (np.pi * length ** 4)
    return CDA


def mach_crit_Korn(
        CL,
        t_over_c,
        sweep=0,
        kappa_A=0.95
):
    """
        Wave drag_force coefficient prediction using the low-fidelity Korn Equation method;
    derived in "Configuration Aerodynamics" by W.H. Mason, Sect. 7.5.2, pg. 7-18

    Args:
        CL: Sectional lift coefficient
        t_over_c: thickness-to-chord ratio
        sweep: sweep angle, in degrees
        kappa_A: Airfoil technology factor (0.95 for supercritical section, 0.87 for NACA 6-series)

    Returns:

    """
    smooth_abs_CL = np.softmax(CL, -CL, hardness=10)

    M_dd = kappa_A / np.cosd(sweep) - t_over_c / np.cosd(sweep) ** 2 - smooth_abs_CL / (10 * np.cosd(sweep) ** 3)
    M_crit = M_dd - (0.1 / 80) ** (1 / 3)
    return M_crit


def Cd_wave_Korn(Cl, t_over_c, mach, sweep=0, kappa_A=0.95):
    """
    Wave drag_force coefficient prediction using the low-fidelity Korn Equation method;
    derived in "Configuration Aerodynamics" by W.H. Mason, Sect. 7.5.2, pg. 7-18

    :param Cl: Sectional lift coefficient
    :param t_over_c: thickness-to-chord ratio
    :param sweep: sweep angle, in degrees
    :param kappa_A: Airfoil technology factor (0.95 for supercritical section, 0.87 for NACA 6-series)
    :return: Wave drag coefficient
    """
    smooth_abs_Cl = np.softmax(Cl, -Cl, hardness=10)

    mach = np.fmax(mach, 0)
    Mdd = kappa_A / np.cosd(sweep) - t_over_c / np.cosd(sweep) ** 2 - smooth_abs_Cl / (10 * np.cosd(sweep) ** 3)
    Mcrit = Mdd - (0.1 / 80) ** (1 / 3)
    Cd_wave = np.where(
        mach > Mcrit,
        20 * (mach - Mcrit) ** 4,
        0
    )

    return Cd_wave


def approximate_CD_wave(
        mach,
        mach_crit,
        CD_wave_at_fully_supersonic,
):
    """
    An approximate relation for computing transonic wave drag, based on an object's Mach number.

    Considered reasonably valid from Mach 0 up to around Mach 2 or 3-ish.

    Methodology is a combination of:

        * The methodology described in Raymer, "Aircraft Design: A Conceptual Approach", Section 12.5.10 Transonic Parasite Drag (pg. 449 in Ed. 2)

        and

        * The methodology described in W.H. Mason's Configuration Aerodynamics, Chapter 7. Transonic Aerodynamics of Airfoils and Wings.

    Args:

        mach: Mach number at the operating point to be evaluated

        mach_crit: Critical mach number, a function of the body geometry

        CD_wave_at_fully_supersonic: The wave drag coefficient of the body at the speed that it first goes (
        effectively) fully supersonic.

            Here, that is taken to mean at the Mach 1.2 case.

            This value should probably be derived using something similar to a Sears-Haack relation for the body in
            question, with a markup depending on geometry smoothness.

            The CD_wave predicted by this function will match this value exactly at M=1.2 and M=1.05.

            The peak CD_wave that is predicted is ~1.23 * this value, which occurs at M=1.10.

            In the high-Mach limit, this function asymptotes at 0.80 * this value, as empirically stated by Raymer.
            However, this model is only approximate and is likely not valid for high-supersonic flows.

    Returns: The approximate wave drag coefficient at the specified Mach number.

        The reference area is whatever the reference area used in the `CD_wave_at_fully_supersonic` parameter is.

    """
    mach_crit_max = 1 - (0.1 / 80) ** (1 / 3)

    mach_crit = -np.softmax(
        -mach_crit,
        -mach_crit_max,
        hardness=50
    )

    ### The following approximate relation is derived in W.H. Mason, "Configuration Aerodynamics", Chapter 7. Transonic Aerodynamics of Airfoils and Wings.
    ### Equation 7-8 on Page 7-19.
    ### This is in turn based on Lock's proposed empirically-derived shape of the drag rise, from Hilton, W.F., High Speed Aerodynamics, Longmans, Green & Co., London, 1952, pp. 47-49
    mach_dd = mach_crit + (0.1 / 80) ** (1 / 3)

    ### Model drag sections and cutoffs:
    return CD_wave_at_fully_supersonic * np.where(
        mach < mach_crit,
        0,
        np.where(
            mach < mach_dd,
            20 * (mach - mach_crit) ** 4,
            np.where(
                mach < 1.05,
                cubic_hermite_patch(
                    mach,
                    x_a=mach_dd,
                    x_b=1.05,
                    f_a=20 * (0.1 / 80) ** (4 / 3),
                    f_b=1,
                    dfdx_a=0.1,
                    dfdx_b=8
                ),
                np.where(
                    mach < 1.2,
                    cubic_hermite_patch(
                        mach,
                        x_a=1.05,
                        x_b=1.2,
                        f_a=1,
                        f_b=1,
                        dfdx_a=8,
                        dfdx_b=-4
                    ),
                    np.blend(
                        switch=4 * 2 * (mach - 1.2) / (1.2 - 0.8),
                        value_switch_high=0.8,
                        value_switch_low=1.2,
                    )
                    # 0.8 + 0.2 * np.exp(20 * (1.2 - mach))
                )
            )
        )
    )


if __name__ == '__main__':
    mc = 0.6
    drag = lambda mach: approximate_CD_wave(
        mach,
        mach_crit=mc,
        CD_wave_at_fully_supersonic=1,
    )

    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    fig, ax = plt.subplots(1, 3, figsize=(10, 5))

    mach = np.linspace(0., 2, 10000)
    drag = drag(mach)
    ddragdm = np.gradient(drag, np.diff(mach)[0])
    dddragdm = np.gradient(ddragdm, np.diff(mach)[0])

    plt.sca(ax[0])
    plt.title("$C_D$")
    plt.ylabel("$C_{D, wave} / C_{D, wave, M=1.2}$")
    plt.plot(mach, drag)
    plt.ylim(-0.05, 1.5)

    # plt.ylim(-0.01, 0.05)

    plt.sca(ax[1])
    plt.title("$d(C_D)/d(M)$")
    plt.ylabel(r"$\frac{d(C_{D, wave})}{dM}$")
    plt.plot(mach, ddragdm)
    plt.ylim(-5, 15)

    plt.sca(ax[2])
    plt.title("$d^2(C_D)/d(M)^2$")
    plt.ylabel(r"$\frac{d^2(C_{D, wave})}{dM^2}$")
    plt.plot(mach, dddragdm)
    # plt.ylim(-5, 15)

    for a in ax:
        plt.sca(a)
        plt.xlim(0.6, 1.2)
        plt.xlabel("Mach [-]")

    p.show_plot()
