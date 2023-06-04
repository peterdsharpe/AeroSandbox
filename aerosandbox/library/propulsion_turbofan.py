import aerosandbox.numpy as np


def thrust_turbofan(
        mass_turbofan: float,
) -> float:
    """
    Estimates the maximum rated dry thrust of a turbofan engine. A regression to historical data.

    Based on data for both civilian and military turbofans, available in:
    `aerosandbox/library/datasets/turbine_engines/data.xlsx`

    Applicable to both turbojets and turbofans, and with sizes ranging from micro-turbines (<1 kg) to large transport
    aircraft turbofans.

    See studies in `/AeroSandbox/studies/TurbofanStudies/make_fit_thrust.py` for model details.

    Args:

        mass_turbofan: The mass of the turbofan engine. [kg]

    Returns:

        The maximum (rated takeoff) dry thrust of the turbofan engine. [N]
    """
    p = {'a': 12050.719283568596, 'w': 0.9353861810025565}

    return (
            p["a"] * mass_turbofan ** p["w"]
    )


def thrust_specific_fuel_consumption_turbofan(
        mass_turbofan: float,
        bypass_ratio: float,
) -> float:
    """
    Estimates the thrust-specific fuel consumption (TSFC) of a turbofan engine. A regression to historical data.

    Based on data for both civilian and military turbofans, available in:
    `aerosandbox/library/datasets/turbine_engines/data.xlsx`

    Applicable to both turbojets and turbofans, and with sizes ranging from micro-turbines (<1 kg) to large transport
    aircraft turbofans.

    See studies in `/AeroSandbox/studies/TurbofanStudies/make_fit_tsfc.py` for model details.

    """
    p = {'a'   : 3.2916082331121034e-05, 'Weight [kg]': -0.07792863839756586, 'BPR': -0.3438158689838915,
         'BPR2': 0.29880079602955967}

    return (
            p["a"]
            * mass_turbofan ** p["Weight [kg]"]
            * (bypass_ratio + p["BPR2"]) ** p["BPR"]
    )


def mass_turbofan(
        m_dot_core_corrected,
        overall_pressure_ratio,
        bypass_ratio,
        diameter_fan,

):
    """
    Computes the combined mass of a bare turbofan, nacelle, and accessory and pylon weights.

    Bare weight depends on m_dot, OPR, and BPR.

    Nacelle weight is a function of various areas and fan diameter.

    From TASOPT documentation by Mark Drela, available here: http://web.mit.edu/drela/Public/web/tasopt/TASOPT_doc.pdf
        Section: "Turbofan Weight Model from Historical Data"

    Args:

        m_dot_core_corrected: The mass flow of the core only, corrected to standard conditions. [kg/s]

        overall_pressure_ratio: The overall pressure ratio (OPR) [-]

        bypass_ratio: The bypass ratio (BPR) [-]

        diameter_fan: The diameter of the fan. [m]

    Returns: The total engine mass. [kg]

    """
    kg_to_lbm = 2.20462262
    m_to_ft = 1 / 0.3048

    ##### Compute bare turbofan weight
    m_dot_core_corrected_lbm_per_sec = m_dot_core_corrected * kg_to_lbm  # Converts from kg/s to lbm/s

    ### Parameters determined via least-squares fitting by Drela in TASOPT doc.
    b_m = 1
    b_pi = 1
    b_alpha = 1.2
    W_0_lbm = 1684.5
    W_pi_lbm = 17.7
    W_alpha_lbm = 1662.2

    W_bare_lbm = (
                         m_dot_core_corrected_lbm_per_sec / 100
                 ) ** b_m * (
                         W_0_lbm +
                         W_pi_lbm * (overall_pressure_ratio / 30) ** b_pi +
                         W_alpha_lbm * (bypass_ratio / 5) ** b_alpha
                 )
    W_bare = W_bare_lbm / kg_to_lbm

    ##### Compute nacelle weight

    ### Nondimensional parameters, given by Drela in TASOPT doc.
    r_s_nace = 12
    f_inlet = 0.4
    f_fan = 0.2
    f_exit = 0.4
    r_core = 12

    ### Fan size in imperial units
    d_fan_ft = diameter_fan * m_to_ft
    d_fan_in = d_fan_ft * 12

    ### Compute the diameter of the LPC based on fan diameter and BPR.
    d_LPC_ft = d_fan_ft * (bypass_ratio) ** -0.5

    ### Models from Drela in TASOPT
    S_nace_sqft = r_s_nace * np.pi * (d_fan_ft / 2) ** 2

    A_inlet_sqft = f_inlet * S_nace_sqft
    A_fan_sqft = f_fan * S_nace_sqft
    A_exit_sqft = f_exit * S_nace_sqft
    A_core_sqft = r_core * np.pi * (d_LPC_ft / 2) ** 2

    W_inlet_lbm = A_inlet_sqft * (2.5 + 0.0238 * d_fan_in)
    W_fan_lbm = A_fan_sqft * 1.9
    W_exit_lbm = A_exit_sqft * (2.5 * 0.0363 * d_fan_in)
    W_core_lbm = A_core_sqft * 1.9

    W_nace_lbm = W_inlet_lbm + W_fan_lbm + W_exit_lbm + W_core_lbm
    W_nace = W_nace_lbm / kg_to_lbm

    ##### Compute accessory and pylon weights

    ### Nondimensional parameters, given by Drela in TASOPT doc
    f_add = 0.10
    f_pylon = 0.10

    W_add = f_add * W_bare
    W_pylon = f_pylon * (W_bare + W_add + W_nace)

    ##### Compute the total weight
    W_engine = W_bare + W_add + W_nace + W_pylon

    return W_engine


def m_dot_corrected_over_m_dot(
        temperature_total_2,
        pressure_total_2,
):
    """
    Computes the ratio `m_dot_corrected / m_dot`, where:

        * `m_dot_corrected` is the corrected mass flow rate, where corrected refers to correction to ISO 3977 standard
        temperature and pressure conditions (15C, 101325 Pa).

        * `m_dot` is the raw mass flow rate, at some other conditions.

    Args:

        temperature_total_2: The total temperature at the compressor inlet face, at the conditions to be evaluated. [K]

        pressure_total_2: The total pressure at the compressor inlet face, at the conditions to be evaluated. [Pa]

    Returns:

        The ratio `m_dot_corrected / m_dot`.

    """
    temperature_standard = 273.15 + 15
    pressure_standard = 101325
    return (
            temperature_total_2 / temperature_standard
    ) ** 0.5 / (pressure_total_2 / pressure_standard)


if __name__ == '__main__':
    import aerosandbox as asb

    atmo = asb.Atmosphere(altitude=10668)
    op_point = asb.OperatingPoint(atmo, velocity=0.80 * atmo.speed_of_sound())
    m_dot_corrected_over_m_dot_ratio = m_dot_corrected_over_m_dot(
        temperature_total_2=op_point.total_temperature(),
        pressure_total_2=op_point.total_pressure()
    )

    ### CFM56-2 engine test
    mass_cfm56_2 = mass_turbofan(  # Data here from Wikipedia, cross-referenced to other sources for sanity check.
        m_dot_core_corrected=364 / (5.95 + 1),
        overall_pressure_ratio=31.2,
        bypass_ratio=5.95,
        diameter_fan=1.73
    )  # real mass: (2139 to 2200 kg bare, ~3400 kg installed)
