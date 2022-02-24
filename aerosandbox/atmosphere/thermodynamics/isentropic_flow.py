import aerosandbox.numpy as np


def temperature_over_total_temperature(
        mach,
        gamma=1.4
):
    """
    Gives T/T_t, the ratio of static temperature to total temperature.

    Args:
        mach: Mach number [-]
        gamma: The ratio of specific heats. 1.4 for air across most temperature ranges of interest.
    """
    return (1 + (gamma - 1) / 2 * mach ** 2) ** -1


def pressure_over_total_pressure(
        mach,
        gamma=1.4
):
    """
    Gives P/P_t, the ratio of static pressure to total pressure.

    Args:
        mach: Mach number [-]
        gamma: The ratio of specific heats. 1.4 for air across most temperature ranges of interest.
    """
    return temperature_over_total_temperature(mach=mach, gamma=gamma) ** (gamma / (gamma - 1))


def density_over_total_density(
        mach,
        gamma=1.4
):
    """
    Gives rho/rho_t, the ratio of density to density after isentropic compression.

    Args:
        mach: Mach number [-]
        gamma: The ratio of specific heats. 1.4 for air across most temperature ranges of interest.
    """
    return temperature_over_total_temperature(mach=mach, gamma=gamma) ** (1 / (gamma - 1))


def area_over_choked_area(
        mach,
        gamma=1.4
):
    """
    Gives A/A^* (where A^* is "A-star"), the ratio of cross-sectional flow area to the cross-sectional flow area that would result in choked (M=1) flow.

    Applicable to 1D isentropic nozzle flow.

    Args:
        mach: Mach number [-]
        gamma: The ratio of specific heats. 1.4 for air across most temperature ranges of interest.
    """
    gp1 = gamma + 1
    gm1 = gamma - 1

    return (
            (gp1 / 2) ** (-gp1 / (2 * gm1)) *
            (1 + gm1 / 2 * mach ** 2) ** (gp1 / (2 * gm1)) / mach
    )


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    mach = np.linspace(0.5, 3, 500)

    fig, ax = plt.subplots()

    for name, data in {
        "$T/T_t$"       : temperature_over_total_temperature(mach),
        "$P/P_t$"       : pressure_over_total_pressure(mach),
        "$A/A^*$"       : area_over_choked_area(mach),
        r"$\rho/\rho_t$": density_over_total_density(mach),
    }.items():
        plt.plot(mach, data, label=name)
    p.show_plot()
