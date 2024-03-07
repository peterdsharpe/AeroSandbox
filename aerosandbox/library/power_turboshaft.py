import aerosandbox.numpy as np


def overall_pressure_ratio_turboshaft_technology_limit(
        mass_turboshaft: float
) -> float:
    """
    Estimates the maximum-practically-achievable overall pressure ratio (OPR) of a turboshaft engine, as a function
    of its mass. A regression to historical data.

    Based on an envelope of data for both civilian and military turboshafts (including RC-scale turboshafts), available in:
    `aerosandbox/library/datasets/turbine_engines/data.xlsx`

    See study in `/AeroSandbox/studies/TurboshaftStudies/make_fit_overall_pressure_ratio.py` for model details.

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
    Estimates the maximum rated power of a turboshaft engine, given its mass. A regression to historical data.

    Based on data for both civilian and military turboshafts, available in:
    `aerosandbox/library/datasets/turbine_engines/data.xlsx`

    See studies in `/AeroSandbox/studies/TurboshaftStudies/make_fit_power.py` for model details.

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
    Estimates the thermal efficiency of a turboshaft engine. A regression to historical data.

    Based on data for both civilian and military turboshafts, available in:
    `aerosandbox/library/datasets/turbine_engines/data.xlsx`

    See studies in `/AeroSandbox/studies/TurboshaftStudies/make_turboshaft_fits.py` for model details.

    Thermal efficiency knockdown at partial power is based on:
        Ingmar Geiß, "Sizing of the Series Hybrid-electric Propulsion System of General Aviation Aircraft", 2020.
        PhD Thesis, University of Stuttgart. Page 18, Figure 3.2.

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

    thermal_efficiency_at_full_power = np.blend(
        p["a"] + (np.log10(mass_turboshaft) - p["wcen"]) / p["wscl"],
        value_switch_high=ideal_efficiency,
        value_switch_low=0,
    )

    p = {
        'B0': 0.0592,  # Modified from Geiß thesis such that B values sum to 1 by construction. Orig: 0.05658
        'B1': 2.567,
        'B2': -2.612,
        'B3': 0.9858
    }

    thermal_efficiency_knockdown_from_partial_power = (
            p["B0"]
            + p["B1"] * throttle_setting
            + p["B2"] * throttle_setting ** 2
            + p["B3"] * throttle_setting ** 3
    )

    return (
            thermal_efficiency_at_full_power
            * thermal_efficiency_knockdown_from_partial_power
    )


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    fig, ax = plt.subplots()
    x = np.linspace(0, 1)
    plt.plot(
        x,
        thermal_efficiency_turboshaft(1000, throttle_setting=x) / thermal_efficiency_turboshaft(1000),
    )
    ax.xaxis.set_major_formatter(p.mpl.ticker.PercentFormatter(1))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    p.set_ticks(0.1, 0.05, 0.1, 0.05)

    p.show_plot(
        "Turboshaft: Thermal Efficiency at Partial Power",
        "Throttle Setting [-]",
        "Thermal Efficiency Knockdown relative to Design Point [-] $\eta / \eta_\mathrm{max}$"
    )

    ##### Do Weight/OPR Efficiency Plot #####

    fig, ax = plt.subplots()
    mass = np.geomspace(1e0, 1e4, 300)
    opr = np.geomspace(1, 100, 500)

    Mass, Opr = np.meshgrid(mass, opr)

    Mask = overall_pressure_ratio_turboshaft_technology_limit(Mass) > Opr

    cont, contf, cbar = p.contour(
        Mass,
        Opr,
        thermal_efficiency_turboshaft(Mass, Opr),
        mask=Mask,
        linelabels_format=lambda x: f"{x:.0%}",
        x_log_scale=True,
        colorbar_label="Thermal Efficiency [%]",
        cmap="turbo_r",
    )

    cbar.ax.yaxis.set_major_formatter(p.mpl.ticker.PercentFormatter(1, decimals=0))

    p.set_ticks(None, None, 5, 1)

    p.show_plot(
        "Turboshaft Model: Thermal Efficiency vs. Weight and OPR",
        "Engine Weight [kg]",
        "Overall Pressure Ratio [-]",
        dpi=300
    )
