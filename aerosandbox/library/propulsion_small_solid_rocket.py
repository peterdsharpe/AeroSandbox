import aerosandbox.numpy as np

# import firefly_propulsion.propellant.oxamide_model as oxm
# from proptools import nozzle as ptn

# zeta_c_star = oxm.zeta_c_star
# chamber_pressure_max = oxm.p_c_max
# n = oxm.n


# Burn rate exponent [units: dimensionless]
# Value of 0.402 based on strand burner fit as of 2020-02-21
n = 0.402

# Oxamide parameter [units: dimensionless].
# Theoretically, lambda = 13.3 for a 80% AP + 20% HTPB propellant,
# which we have used as the basis for the no-metal propellant family
# since spring 2018.
# Previously, (e.g. in Matt's MS Thesis) we used a metalized propellant
# which had a lower theoretical value for lambda (~7).
# lamb = 13.3
# Value of 6.20 based on strand burner fit as of 2020-02-21
lamb = 6.20

# Burn rate coefficient at zero oxamide content
# [units: pascal**(-n) meter second**-1].
# This is for propellant in the motor.
# Based on strand burner fit for 400 um blend AP as of 2020-02-21
a_0 = 3.43 * 1.15 * (1e6) ** (-n) * 1e-3

# Strand burner burn rate reduction factor
strand_reduction_factor = 1 / 1.15

# Combustion efficiency [units: dimensionless].
zeta_c_star = 0.90

# Maximum chamber pressure the burn rate model is fit to [units: pascal].
chamber_pressure_max = 2.0e6

# Valid range of oxamide mass fraction values for the model.
# [units: dimensionless]
W_OM_VALID_RANGE = (0, 0.22)
OUT_OF_RANGE_ERROR_STRING = (
    '{:.3f} is outside the model valid range of {:.3f} <= w_om <= {:.3f}')


def burn_rate_coefficient(oxamide_fraction):
    """Burn rate vs oxamide content model.
    Valid from 0% to 15% oxamide. # TODO IMPLEMENT THIS

    Returns:
        a: propellant burn rate coefficient
            [units: pascal**(-n) meter second**-1].
    """
    oxamide_fraction = np.fmax(oxamide_fraction, 0)

    return a_0 * (1 - oxamide_fraction) / (1 + lamb * oxamide_fraction)


def c_star(oxamide_fraction):
    """Characteristic velocity vs. oxamide content model.
    Valid from 0% to 20% oxamide. # TODO IMPLEMENT THIS

    Returns:
        c_star: ideal characteristic velocity [units: meter second**-1].
    """
    # oxamide_fraction = cas.fmax(oxamide_fraction, 0)
    coefs = [1380.2, -983.3, -697.1]
    return coefs[0] + coefs[1] * oxamide_fraction + coefs[2] * oxamide_fraction ** 2


def dubious_min_combustion_pressure(oxamide_fraction):
    """Minimum pressure for stable combustion vs. oxamide content model.

    Note: this model is of DUBIOUS accuracy. Don't trust it.
    """
    coefs = [7.73179444e+00, 3.60886970e-01, 7.64587936e-03]
    p_min_MPa = coefs[0] * oxamide_fraction ** 2 + coefs[1] * oxamide_fraction + coefs[2]
    p_min = 1e6 * p_min_MPa
    return p_min  # Pa


def gamma(oxamide_fraction):
    """Ratio of specific heats vs. oxamide content model.

    Returns:
        gamma: Exhaust gas ratio of specific heats [units: dimensionless].
    """
    # oxamide_fraction = cas.fmax(oxamide_fraction, 0)

    coefs = [1.238, 0.216, -0.351]
    return coefs[0] + coefs[1] * oxamide_fraction + coefs[2] * oxamide_fraction ** 2


def expansion_ratio_from_pressure(chamber_pressure, exit_pressure, gamma, oxamide_fraction):
    """Find the nozzle expansion ratio from the chamber and exit pressures.

    See :ref:`expansion-ratio-tutorial-label` for a physical description of the
    expansion ratio.

    Reference: Rocket Propulsion Elements, 8th Edition, Equation 3-25

    Arguments:
        chamber_pressure (scalar): Nozzle stagnation chamber pressure [units: pascal].
        exit_pressure (scalar): Nozzle exit static pressure [units: pascal].
        gamma (scalar): Exhaust gas ratio of specific heats [units: dimensionless].

    Returns:
        scalar: Expansion ratio :math:`\\epsilon = A_e / A_t` [units: dimensionless]
    """
    chamber_pressure = np.fmax(chamber_pressure, dubious_min_combustion_pressure(oxamide_fraction))
    chamber_pressure = np.fmax(chamber_pressure, exit_pressure * 1.5)

    AtAe = ((gamma + 1) / 2) ** (1 / (gamma - 1)) \
           * (exit_pressure / chamber_pressure) ** (1 / gamma) \
           * np.sqrt((gamma + 1) / (gamma - 1) * (1 - (exit_pressure / chamber_pressure) ** ((gamma - 1) / gamma)))
    er = 1 / AtAe
    return er


def thrust_coefficient(chamber_pressure, exit_pressure, gamma, p_a=None, er=None):
    """Nozzle thrust coefficient, :math:`C_F`.

    The thrust coefficient is a figure of merit for the nozzle expansion process.
    See :ref:`thrust-coefficient-label` for a description of the physical meaning of the
    thrust coefficient.

    Reference: Equation 1-33a in Huzel and Huang.

    Arguments:
        chamber_pressure (scalar): Nozzle stagnation chamber pressure [units: pascal].
        exit_pressure (scalar): Nozzle exit static pressure [units: pascal].
        gamma (scalar): Exhaust gas ratio of specific heats [units: dimensionless].
        p_a (scalar, optional): Ambient pressure [units: pascal]. If None,
            then p_a = exit_pressure.
        er (scalar, optional): Nozzle area expansion ratio [units: dimensionless]. If None,
            then p_a = exit_pressure.

    Returns:
        scalar: The thrust coefficient, :math:`C_F` [units: dimensionless].
    """
    # if (p_a is None and er is not None) or (er is None and p_a is not None):
    #     raise ValueError('Both p_a and er must be provided.')
    C_F = (2 * gamma ** 2 / (gamma - 1)
           * (2 / (gamma + 1)) ** ((gamma + 1) / (gamma - 1))
           * (1 - (exit_pressure / chamber_pressure) ** ((gamma - 1) / gamma))
           ) ** 0.5
    # if p_a is not None and er is not None:
    C_F += er * (exit_pressure - p_a) / chamber_pressure

    return C_F


if __name__ == "__main__":
    import plotly.express as px
    import pandas as pd

    # Oxamide Function tests
    oxamides = np.linspace(-0.3, 0.5, 200)
    burn_rate_coefficients = burn_rate_coefficient(oxamides)
    c_stars = c_star(oxamides)
    min_combustion_pressures = dubious_min_combustion_pressure(oxamides)
    gammas = gamma(oxamides)
    px.scatter(x=oxamides, y=burn_rate_coefficients, labels={"x": "Oxamide", "y": "Burn Rate Coeff"}).show()
    px.scatter(x=oxamides, y=c_stars, labels={"x": "Oxamide", "y": "c_star"}).show()
    px.scatter(x=oxamides, y=min_combustion_pressures, labels={"x": "Oxamide", "y": "Min. Combustion Pressure"}).show()
    px.scatter(x=oxamides, y=gammas, labels={"x": "Oxamide", "y": "Gamma"}).show()

    # # ER_from_P test
    chamber_pressure_inputs = np.logspace(5, 6, 200)
    exit_pressure_inputs = np.logspace(4, 5, 200)
    ox_for_test = 0
    chamber_pressures = []
    exit_pressures = []
    ers = []
    for chamber_pressure in chamber_pressure_inputs:
        for exit_pressure in exit_pressure_inputs:
            chamber_pressures.append(chamber_pressure)
            exit_pressures.append(exit_pressure)
            ers.append(expansion_ratio_from_pressure(chamber_pressure, exit_pressure, gamma(ox_for_test), ox_for_test))
    data = pd.DataFrame({
        'chamber_pressure': chamber_pressures,
        'exit_pressure'   : exit_pressures,
        'ers'             : ers
    })
    px.scatter_3d(data, x='chamber_pressure', y='exit_pressure', z='ers', color='ers', log_x=True, log_y=True,
                  log_z=True).show()
