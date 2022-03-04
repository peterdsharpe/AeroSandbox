import aerosandbox.numpy as np
from aerosandbox.atmosphere.thermodynamics.gas import universal_gas_constant


def mass_flow_rate(
        mach,
        area,
        total_pressure,
        total_temperature,
        molecular_mass=28.9644e-3,
        gamma=1.4,
):
    specific_gas_constant = universal_gas_constant / molecular_mass
    return (
            (area * total_pressure) * (gamma / specific_gas_constant / total_temperature) ** 0.5
            * mach * (1 + (gamma - 1) / 2 * mach ** 2) ** (- (gamma + 1) / (2 * (gamma - 1)))
    )
