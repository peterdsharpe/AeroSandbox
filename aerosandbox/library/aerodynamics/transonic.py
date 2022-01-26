import aerosandbox.numpy as np
from aerosandbox.modeling.splines.hermite import linear_hermite_patch, cubic_hermite_patch


def sears_haack_drag(radius_max: float, length: float) -> float:
    """
    Yields the idealized drag area (denoted CDA, or equivalently, D/q) of a Sears-Haack body.

    Assumes linearized supersonic (Prandtl-Glauert) flow.

    https://en.wikipedia.org/wiki/Sears%E2%80%93Haack_body

    Args:
        radius_max: The maximum radius of the Sears-Haack body.
        length: The length of the Sears-Haack body.

    Returns: The drag area (CDA, or D/q) of the body. To get the drag force, multiply by the dynamic pressure.

    """
    CDA = 9 * np.pi ** 2 * radius_max ** 2 / (2 * length ** 2)
    return CDA


def approximate_CD_wave(
        mach,
        mach_crit,
        CD_wave_at_fully_supersonic,
):
    """
    An approximate relation for computing transonic wave drag, based on an object's Mach number.

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

    return CD_wave_at_fully_supersonic * np.where(
        mach < mach_crit,
        0,
        np.where(
            mach < mach_dd,
            20 * (mach - mach_crit) ** 4,
            np.where(
                mach < 1,
                np.exp(cubic_hermite_patch(
                    np.clip(mach, mach_dd - 0.001, 1.051),
                    x_a=mach_dd,
                    x_b=1,
                    f_a=np.log(20 * (0.1 / 80) ** (4 / 3)),
                    f_b=np.log(0.5),
                    dfdx_a=0.1 / (20 * (0.1 / 80) ** (4 / 3)),
                    dfdx_b=10 / (0.5)
                )),
                np.where(
                    mach < 1.05,
                    linear_hermite_patch(
                        x=mach,
                        x_a=1,
                        x_b=1.05,
                        f_a=0.5,
                        f_b=1
                    ),
                    np.where(
                        mach < 1.2,
                        cubic_hermite_patch(
                            mach,
                            x_a=1.05,
                            x_b=1.2,
                            f_a=1,
                            f_b=1,
                            dfdx_a=10,
                            dfdx_b=-0.2 * 2
                        ),
                        0.8 + 0.2 * np.exp(2 * (1.2 - mach))
                    )
                )
            )
        )
    )


if __name__ == '__main__':
    mc = 0.7
    d = lambda mach: approximate_CD_wave(mach, mach_crit=mc, CD_wave_at_fully_supersonic=1)

    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    fig, ax = plt.subplots(2, 1)

    m = np.linspace(0., 2, 1000)
    plt.sca(ax[0])
    plt.ylabel("$C_{D, wave} / C_{D, wave, M=1.2}$")
    plt.plot(m, d(m))
    for mi in [mc, 1, 1.05, 1.2, mc + (0.1 / 80) ** (1 / 3)]:
        plt.plot([mi], [d(mi)], ".k")
    # plt.ylim(0, 0.005)
    plt.sca(ax[1])
    plt.ylabel(r"$\frac{d(C_{D, wave})}{dM}$")
    plt.plot(np.trapz(m), np.diff(d(m)) / np.diff(m))
    plt.ylim(-5, 15)
    p.show_plot(
        None,
        "Mach [-]"
    )