def mass_gas_engine(max_power):
    """
    Estimates the mass of a small piston-driven motor.
    Source: https://docs.google.com/spreadsheets/d/103VPDwbQ5PfIE3oQl4CXxM5AP6Ueha-zbw7urElkQBM/edit#gid=0
    :param max_power: Maximum power output [W]
    :return: Estimated motor mass [kg]
    """
    max_power_hp = max_power / 745.7
    mass_lbm = 6.12 * max_power_hp ** 0.588
    mass = mass_lbm * 0.453592 # to kilograms

    return mass