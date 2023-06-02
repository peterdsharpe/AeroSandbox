import aerosandbox.numpy as np


def overall_pressure_ratio_turboshaft_technology_limit(
        mass_turboshaft: float
) -> float:
    """
    Estimates the maximum-practically-achievable overall pressure ratio (OPR) of a turboshaft engine, as a function
    of its mass.

    Based on an envelope of data for both civilian and military turboshafts (including RC-scale turboshafts), available in:
    `aerosandbox/library/datasets/turbine_engines/data.xlsx`

    See study in `/AeroSandbox/studies/TurbineStudies/make_fit_overall_pressure_ratio.py` for model details.

    Args:

        mass_turboshaft: The mass of the turboshaft engine. [kg]

    Returns:

        The maximum-practically-achievable overall pressure ratio (OPR) of the turboshaft engine. [-]
    """
    p = {'scl': 1.0222956615376533, 'cen': 1.6535195257959798, 'high': 23.957335474997656}
    return np.blend(
        np.log10(mass_turboshaft) / p["scl"] - p["cen"],
        value_switch_high=p["high"],
        value_switch_low=1,
    )


def power_turboshaft(
        mass_turboshaft: float,
        overall_pressure_ratio: float = None,
) -> float:
    """
    Estimates the maximum rated power of a turboshaft engine.

    Based on data for both civilian and military turboshafts, available in:
    `aerosandbox/library/datasets/turbine_engines/data.xlsx`

    See studies in `/AeroSandbox/studies/TurbineStudies/make_fit_power.py` for model details.

    Args:

        mass_turboshaft: The mass of the turboshaft engine. [kg]

        overall_pressure_ratio: The overall pressure ratio of the turboshaft engine. [-] If unspecified, a sensible
            default based on the technology envelope (with a 0.7x knockdown) will be used.

    Returns:

        The maximum (rated takeoff) power of the turboshaft engine. [W]

    """
    if overall_pressure_ratio is None:
        overall_pressure_ratio = overall_pressure_ratio_turboshaft_technology_limit(
            mass_turboshaft
        ) * 0.7

    p = {'a': 1674.9411795202134, 'OPR': 0.5090953411025091, 'Weight [kg]': 0.9418482117552568}
    return (
            p["a"]
            * mass_turboshaft ** p["Weight [kg]"]
            * overall_pressure_ratio ** p["OPR"]
    )


def thermal_efficiency_turboshaft(
        mass_turboshaft: float,
        overall_pressure_ratio: float = None,
        throttle_setting: float = 1,
) -> float:
    """
    Estimates the thermal efficiency of a turboshaft engine.

    Based on data for both civilian and military turboshafts, available in:
    `aerosandbox/library/datasets/turbine_engines/data.xlsx`

    See studies in `/AeroSandbox/studies/TurbineStudies/make_turboshaft_fits.py` for model details.

    Args:

        mass_turboshaft: The mass of the turboshaft engine. [kg]

        overall_pressure_ratio: The overall pressure ratio of the turboshaft engine. [-] If unspecified, a sensible
            default based on the technology envelope (with a 0.7x knockdown) will be used.

        throttle_setting: The throttle setting of the turboshaft engine. [-] 1 is full throttle, 0 is no throttle.

    Returns:

        The thermal efficiency of the turboshaft engine. [-]

    """
    if overall_pressure_ratio is None:
        overall_pressure_ratio = overall_pressure_ratio_turboshaft_technology_limit(
            mass_turboshaft
        ) * 0.7

    p = {'a': 0.12721246565294902, 'wcen': 2.679474077211383, 'wscl': 4.10824884208311}

    ideal_efficiency = 1 - (1 / overall_pressure_ratio) ** ((1.4 - 1) / 1.4)

    return np.blend(
        p["a"] + (np.log10(mass_turboshaft) - p["wcen"]) / p["wscl"],
        value_switch_high=ideal_efficiency,
        value_switch_low=0,
    )
