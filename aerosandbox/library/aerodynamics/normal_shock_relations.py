# From https://www.grc.nasa.gov/WWW/K-12/airplane/normal.html

def mach_number_after_normal_shock(
        mach_upstream,
        gamma=1.4,
):
    """
    Computes the mach number immediately after a normal shock wave.

    Args:
        mach_upstream: The mach number immediately before the normal shock wave.
        gamma: The ratio of specific heats of the fluid. 1.4 for air.

    Returns: The mach number immediately after the normal shock wave.

    """
    gm1 = gamma - 1
    m2 = mach_upstream ** 2

    return (
            (gm1 * m2 + 2) / (2 * gamma * m2 - gm1)
    ) ** 0.5


def density_ratio_across_normal_shock(
        mach_upstream,
        gamma=1.4
):
    """
    Computes the ratio of fluid density across a normal shock.

    Specifically, returns: rho_after_shock / rho_before_shock

    Args:
        mach_upstream: The mach number immediately before the normal shock wave.
        gamma: The ratio of specific heats of the fluid. 1.4 for air.

    Returns: rho_after_shock / rho_before_shock

    """
    return (
            (gamma + 1) * mach_upstream ** 2
    ) / (
            (gamma - 1) * mach_upstream ** 2 + 2
    )


def temperature_ratio_across_normal_shock(
        mach_upstream,
        gamma=1.4
):
    """
    Computes the ratio of fluid temperature across a normal shock.

    Specifically, returns: T_after_shock / T_before_shock

    Args:
        mach_upstream: The mach number immediately before the normal shock wave.
        gamma: The ratio of specific heats of the fluid. 1.4 for air.

    Returns: T_after_shock / T_before_shock

    """
    gm1 = gamma - 1
    m2 = mach_upstream ** 2
    return (
            (2 * gamma * m2 - gm1) * (gm1 * m2 + 2)
    ) / (
            (gamma + 1) ** 2 * m2
    )


def pressure_ratio_across_normal_shock(
        mach_upstream,
        gamma=1.4
):
    """
    Computes the ratio of fluid static pressure across a normal shock.

    Specifically, returns: P_after_shock / P_before_shock

    Args:
        mach_upstream: The mach number immediately before the normal shock wave.
        gamma: The ratio of specific heats of the fluid. 1.4 for air.

    Returns: P_after_shock / P_before_shock

    """
    m2 = mach_upstream ** 2
    return (
            2 * gamma * m2 - (gamma - 1)
    ) / (
        (gamma + 1)
    )


def total_pressure_ratio_across_normal_shock(
        mach_upstream,
        gamma=1.4
):
    """
    Computes the ratio of fluid total pressure across a normal shock.

    Specifically, returns: Pt_after_shock / Pt_before_shock

    Args:
        mach_upstream: The mach number immediately before the normal shock wave.
        gamma: The ratio of specific heats of the fluid. 1.4 for air.

    Returns: Pt_after_shock / Pt_before_shock

    """
    return density_ratio_across_normal_shock(
        mach_upstream=mach_upstream,
        gamma=gamma
    ) ** (gamma / (gamma - 1)) * (
            (gamma + 1) / (2 * gamma * mach_upstream ** 2 - (gamma - 1))
    ) ** (1 / (gamma - 1))


if __name__ == '__main__':
    def q_ratio(mach):
        return (
                density_ratio_across_normal_shock(mach) *
                (
                        mach_number_after_normal_shock(mach) *
                        temperature_ratio_across_normal_shock(mach) ** 0.5
                ) ** 2
        )


    q_ratio(2)
