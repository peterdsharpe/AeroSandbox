from aerosandbox.atmosphere.thermodynamics.gas import universal_gas_constant


def mass_flow_rate(
    mach,
    area,
    total_pressure,
    total_temperature,
    molecular_mass=28.9644e-3,
    gamma=1.4,
):
    """
    Compute the mass flow rate of a compressible flow through a given cross-sectional area.

    Parameters
    ----------
    mach
        Mach number [-]
    area
        Cross-sectional flow area [m^2]
    total_pressure
        Total (stagnation) pressure of the flow [Pa]
    total_temperature
        Total (stagnation) temperature of the flow [K]
    molecular_mass : float
        Molecular mass of the gas [kg/mol]. The default value is that of air.
    gamma : float
        The ratio of specific heats. 1.4 for air.

    Returns
    -------
    Mass flow rate [kg/s].
    """
    specific_gas_constant = universal_gas_constant / molecular_mass
    return (
        (area * total_pressure)
        * (gamma / specific_gas_constant / total_temperature) ** 0.5
        * mach
        * (1 + (gamma - 1) / 2 * mach**2) ** (-(gamma + 1) / (2 * (gamma - 1)))
    )
