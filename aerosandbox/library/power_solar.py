import aerosandbox.numpy as np
from aerosandbox.atmosphere.atmosphere import Atmosphere
from typing import Union

"""
Welcome to the AeroSandbox solar energy library!

The function you're probably looking for is `solar_flux()`, which summarizes this entire module and computes the 
realized solar flux on a given surface as a function of many different parameters.
"""


def _prepare_for_inverse_trig(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Ensures that a value is within the open interval (-1, 1), so that if you call an inverse trig function
    on it (e.g., arcsin, arccos), you won't get an infinity or NaN.

    Args:
        x: A floating-point number or an array of such. If an array, this function acts elementwise.

    Returns: A clipped version of the number, constrained to be in the open interval (-1, 1).
    """
    return (
            np.nextafter(1, -1) *
            np.clip(x, -1, 1)
    )


def solar_flux_outside_atmosphere_normal(
        day_of_year: Union[int, float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Computes the normal solar flux at the top of the atmosphere ("Airmass 0").
    This varies due to Earth's orbital eccentricity (elliptical orbit).

    Source: https://www.itacanet.org/the-sun-as-a-source-of-energy/part-2-solar-energy-reaching-the-earths-surface/#2.1.-The-Solar-Constant

    Args:
        day_of_year: Julian day (1 == Jan. 1, 365 == Dec. 31)

    Returns: The normal solar flux [W/m^2] at the top of the atmosphere.

    """
    return 1367 * (
            1 + 0.034 * np.cosd(360 * (day_of_year) / 365.25)
    )


def declination_angle(
        day_of_year: Union[int, float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Computes the solar declination angle, in degrees, as a function of day of year.

    Accounts for the Earth's obliquity.

    Source: https://www.pveducation.org/pvcdrom/properties-of-sunlight/declination-angle

    Args:
        day_of_year: Julian day (1 == Jan. 1, 365 == Dec. 31)

    Returns: Solar declination angle [deg]

    """
    return -23.4398 * np.cosd(360 / 365.25 * (day_of_year + 10))


def solar_elevation_angle(
        latitude: Union[float, np.ndarray],
        day_of_year: Union[int, float, np.ndarray],
        time: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Elevation angle of the sun [degrees] for a local observer.

    Solar elevation angle is the angle between the Sun's position and the local horizon plane.
    (Solar elevation angle) = 90 deg - (solar zenith angle)

    Source: https://www.pveducation.org/pvcdrom/properties-of-sunlight/elevation-angle

    Args:
        latitude: Local geographic latitude [degrees]. Positive for north, negative for south.
        day_of_year: Julian day (1 == Jan. 1, 365 == Dec. 31)
        time: Time after local solar noon [seconds]

    Returns: Solar elevation angle [degrees] (angle between horizon and sun). Returns negative values if the sun is
    below the horizon.

    """
    declination = declination_angle(day_of_year)

    sin_solar_elevation_angle = (
            np.sind(declination) * np.sind(latitude) +
            np.cosd(declination) * np.cosd(latitude) * np.cosd(time / 86400 * 360)
    )

    solar_elevation_angle = np.arcsind(
        _prepare_for_inverse_trig(sin_solar_elevation_angle)
    )  # in degrees
    return solar_elevation_angle


def solar_azimuth_angle(
        latitude: Union[float, np.ndarray],
        day_of_year: Union[int, float, np.ndarray],
        time: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Azimuth angle of the sun [degrees] for a local observer.

    Source: https://www.pveducation.org/pvcdrom/properties-of-sunlight/azimuth-angle

    Args:
        latitude: Local geographic latitude [degrees]. Positive for north, negative for south.
        day_of_year: Julian day (1 == Jan. 1, 365 == Dec. 31)
        time: Time after local solar noon [seconds]

    Returns: Solar azimuth angle [degrees] (the compass direction from which the sunlight is coming).
        * 0 corresponds to North, 90 corresponds to East.
        * Output ranges from 0 to 360 degrees.

    """
    declination = declination_angle(day_of_year)
    sdec = np.sind(declination)
    cdec = np.cosd(declination)
    slat = np.sind(latitude)
    clat = np.cosd(latitude)
    ctime = np.cosd(time / 86400 * 360)

    elevation = solar_elevation_angle(latitude, day_of_year, time)
    cele = np.cosd(elevation)

    cos_azimuth = (sdec * clat - cdec * slat * ctime) / cele

    azimuth_raw = np.arccosd(_prepare_for_inverse_trig(cos_azimuth))

    is_solar_morning = np.mod(time, 86400) > 43200

    solar_azimuth_angle = np.where(
        is_solar_morning,
        azimuth_raw,
        360 - azimuth_raw
    )

    return solar_azimuth_angle


def airmass(
        solar_elevation_angle: Union[float, np.ndarray],
        altitude: Union[float, np.ndarray] = 0.,
        method='Young'
) -> Union[float, np.ndarray]:
    """
    Computes the (relative) airmass as a function of the (true) solar elevation angle and observer altitude.
    Includes refractive (e.g. curving) effects due to atmospheric density gradient.

    Airmass is the line integral of air density along an upwards-pointed ray, extended to infinity. As a raw
    calculation of "absolute airmass", this would have units of kg/m^2. It varies primarily as a function of solar
    elevation angle and observer altitude. (Higher altitude means less optical absorption.) However,
    "airmass" usually refers to the "relative airmass", which is the absolute airmass of a given scenario divided by
    the absolute airmass of a reference scenario. This reference scenario is when the sun is directly overhead (solar
    elevation angle of 90 degrees) and the observer is at sea level.

    Therefore:

        * Outer space has a (relative) airmass of 0 (regardless of solar elevation angle).

        * Sea level with the sun directly overhead has a (relative) airmass of 1.

        * Sea level with the sun at the horizon has a (relative) airmass of ~31.7. (Not infinity, since the Earth is
        spherical, so the ray eventually reaches outer space.) Some models will say that this relative airmass at the
        horizon should be ~38; that is only true if one uses the *apparent* solar elevation angle, rather than the
        *true* (geometric) one. The discrepancy comes from the fact that light refracts (curves) as it passes through
        the atmosphere's density gradient, with the difference between true and apparent elevation angles reaching a
        maximum of 0.56 degrees at the horizon.

    Solar elevation angle is the angle between the Sun's position and the horizon.
    (Solar elevation angle) = 90 deg - (solar zenith angle)

    Note that for small negative values of the solar elevation angle (e.g., -0.5 degree), airmass remains finite,
    due to ray refraction (curving) through the atmosphere.

    For significantly negative values of the solar elevation angle (e.g., -10 degrees), the airmass is theoretically
    infinite. This function returns 1e100 in lieu of this here.

    Sources:

        Young model: Young, A. T. 1994. Air mass and refraction. Applied Optics. 33:1108–1110. doi:
        10.1364/AO.33.001108. Reproduced at https://en.wikipedia.org/wiki/Air_mass_(astronomy)

    Args:

        solar_elevation_angle: Solar elevation angle [degrees] (angle between horizon and sun). Note that we use the
        true solar elevation angle, rather than the apparent one. The discrepancy comes from the fact that light
        refracts (curves) as it passes through the atmosphere's density gradient, with the difference between true
        and apparent elevation angles reaching a maximum of 0.56 degrees at the horizon.

        altitude: Altitude of the observer [meters] above sea level.

        method: A string that determines which model to use.

    Returns: The relative airmass [unitless] as a function of the (true) solar elevation angle and observer altitude.

        * Always ranges from 0 to Infinity

    """
    true_zenith_angle = 90 - solar_elevation_angle

    if method == 'Young':
        cos_zt = np.cosd(true_zenith_angle)
        cos2_zt = cos_zt ** 2
        cos3_zt = cos_zt ** 3

        numerator = 1.002432 * cos2_zt + 0.148386 * cos_zt + 0.0096467
        denominator = cos3_zt + 0.149864 * cos2_zt + 0.0102963 * cos_zt + 0.000303978

        sea_level_airmass = np.where(
            denominator > 0,
            numerator / denominator,
            1e100  # Essentially, infinity.
        )
    else:
        raise ValueError("Bad value of `method`!")

    airmass_at_altitude = sea_level_airmass * (
            Atmosphere(altitude=altitude).pressure() /
            101325.
    )

    return airmass_at_altitude


def solar_flux(
        latitude: Union[float, np.ndarray],
        day_of_year: Union[int, float, np.ndarray],
        time: Union[float, np.ndarray],
        altitude: Union[float, np.ndarray] = 0.,
        panel_azimuth_angle: Union[float, np.ndarray] = 0.,
        panel_tilt_angle: Union[float, np.ndarray] = 0.,
        air_quality: str = 'typical',
        albedo: Union[float, np.ndarray] = 0.2,
        **deprecated_kwargs
) -> Union[float, np.ndarray]:
    """
    Computes the solar power flux (power per unit area) on a flat (possibly tilted) panel. Accounts for atmospheric
    absorption, scattering, and re-scattering (e.g. diffuse illumination), all as a function of panel altitude.

    Fully vectorizable.

    Source for atmospheric absorption:

        * Planning and installing photovoltaic systems: a guide for installers, architects and engineers,
        2nd Ed. (2008), Table 1.1, Earthscan with the International Institute for Environment and Development,
        Deutsche Gesellschaft für Sonnenenergie. ISBN 1-84407-442-0., accessed via
        https://en.wikipedia.org/wiki/Air_mass_(solar_energy)

    Args:

        latitude: Local geographic latitude [degrees]. Positive for north, negative for south.

        day_of_year: The day of the year, represented in the Julian day format (i.e., 1 == Jan. 1, 365 == Dec. 31). This
            accounts for seasonal variations in the sun's position in the sky.

        time: The time of day, measured as the time elapsed after local solar noon [seconds]. Should range from 0 to
            86,400 (the number of seconds in a day). Local solar noon is the time of day when the sun is at its highest
            point in the sky, directly above the observer's local meridian. This is the time when the sun's rays are most
            directly overhead and solar flux is at its peak for a given location. Solar noon does not necessarily occur
            at exactly 12:00 PM local standard time, as it depends on your longitude, the equation of time, and the time
            of year. (Also, don't forget to account for Daylight Savings Time, if that's a relevant consideration for
            your location and season.) Typically, local solar noon is +- 15 minutes from 12:00 PM local standard time.

        altitude: Altitude of the panel above sea level [meters]. This affects atmospheric absorption and scattering
            characteristics.

        panel_azimuth_angle: The azimuth angle of the panel normal [degrees] (the compass direction in which the
            panel normal is tilted). Irrelevant if the panel tilt angle is 0 (e.g., the panel is horizontal).

            * 0 corresponds to North, 90 corresponds to East.

            * Input ranges from 0 to 360 degrees.

        panel_tilt_angle: The angle between the panel normal and vertical (zenith) [degrees].

            * Note: this angle convention is different than the solar elevation angle convention!

            * A horizontal panel has a tilt angle of 0, and a vertical panel has a tilt angle of 90 degrees.

            If the angle between the panel normal and the sun direction is ever more than 90 degrees (e.g. the panel
            is pointed the wrong way), we assume that the panel receives no direct irradiation. (However,
            it may still receive minor amounts of power due to diffuse irradiation from re-scattering.)

        air_quality: Indicates the amount of pollution in the air. A string, one of:

            * 'clean': Pristine atmosphere conditions.
            
            * 'typical': Corresponds to "rural aerosol loading" following ASTM G-173.

            * 'polluted': Urban atmosphere conditions.
            
            Note: in very weird edge cases, a polluted atmosphere can actually result in slightly higher solar flux
            than clean air, due to increased back-scattering. For example, imagine it's near sunset, with the sun in
            the west, and your panel normal vector points east. Increased pollution can, in some edge cases,
            result in enough increased back-scattering (multipathing) that you have a smidge more illumination.

        albedo: The fraction of light that hits the ground that is reflected. Affects illumination from re-scattering
            when panels are tilted. Typical values for general terrestrial surfaces are 0.2, which is the default here.

            * Other values, taken from the Earthscan source (citation above):

                * Grass (July, August): 0.25
                * Lawn: 0.18 - 0.23
                * Dry grass: 0.28 - 0.32
                * Milled fields: 0.26
                * Barren soil: 0.17
                * Gravel: 0.18
                * Clean concrete: 0.30
                * Eroded concrete: 0.20
                * Clean cement: 0.55
                * Asphalt: 0.15
                * Forests: 0.05 - 0.18
                * Sandy areas: 0.10 - 0.25
                * Water: Strongly dependent on incidence angle; 0.05 - 0.22
                * Fresh snow: 0.80 - 0.90
                * Old snow: 0.45 - 0.70

    Returns: The solar power flux [W/m^2] on the panel.

        * Note: does not account for any potential reflectivity of the solar panel's coating itself.

    """
    flux_outside_atmosphere = solar_flux_outside_atmosphere_normal(day_of_year=day_of_year)

    solar_elevation = solar_elevation_angle(latitude, day_of_year, time)
    solar_azimuth = solar_azimuth_angle(latitude, day_of_year, time)

    relative_airmass = airmass(
        solar_elevation_angle=solar_elevation,
        altitude=altitude,
    )

    # Source: "Planning and installing..." Earthscan. Full citation in docstring above.
    if air_quality == 'typical':
        atmospheric_transmission_fraction = 0.70 ** (relative_airmass ** 0.678)
    elif air_quality == 'clean':
        atmospheric_transmission_fraction = 0.76 ** (relative_airmass ** 0.618)
    elif air_quality == 'polluted':
        atmospheric_transmission_fraction = 0.56 ** (relative_airmass ** 0.715)
    else:
        raise ValueError("Bad value of `air_quality`!")

    direct_normal_irradiance = np.where(
        solar_elevation > 0.,
        flux_outside_atmosphere * atmospheric_transmission_fraction,
        0.
    )

    absorption_and_scattering_losses = flux_outside_atmosphere - flux_outside_atmosphere * atmospheric_transmission_fraction

    scattering_losses = absorption_and_scattering_losses * (10. / 28.)
    # Source: https://www.pveducation.org/pvcdrom/properties-of-sunlight/air-mass
    # Indicates that absorption and scattering happen in a 18:10 ratio, at least in AM1.5 conditions. We extrapolate
    # this to all conditions.

    panel_tilt_angle = np.mod(panel_tilt_angle, 360)
    fraction_of_panel_facing_sky = np.where(
        panel_tilt_angle < 180,
        1 - panel_tilt_angle / 180,
        -1 + panel_tilt_angle / 180,
    )

    diffuse_irradiance = scattering_losses * atmospheric_transmission_fraction * (
            fraction_of_panel_facing_sky + albedo * (1 - fraction_of_panel_facing_sky)
    )
    # We assume that the in-scattering (i.e., diffuse irradiance) and the out-scattering (i.e., scattering losses in
    # the direct irradiance calculation) are equal, by argument of approximately parallel incident rays.
    # We further assume that any in-scattering must then once-again go through the absorption / re-scattering process,
    # which is identical to the original atmospheric transmission fraction.

    cosine_of_angle_between_panel_normal_and_sun = (
            np.cosd(solar_elevation) *
            np.sind(panel_tilt_angle) *
            np.cosd(panel_azimuth_angle - solar_azimuth)
            + np.sind(solar_elevation) * np.cosd(panel_tilt_angle)
    )
    cosine_of_angle_between_panel_normal_and_sun = np.fmax(
        cosine_of_angle_between_panel_normal_and_sun,
        0
    )  # Accounts for if you have a downwards-pointing panel while the sun is above you.

    # Source: https://www.pveducation.org/pvcdrom/properties-of-sunlight/arbitrary-orientation-and-tilt
    # Author of this code (Peter Sharpe) has manually verified correctness of this vector math.

    flux_on_panel = (
            direct_normal_irradiance * cosine_of_angle_between_panel_normal_and_sun
            + diffuse_irradiance
    )

    return flux_on_panel


def peak_sun_hours_per_day_on_horizontal(
        latitude: Union[float, np.ndarray],
        day_of_year: Union[int, float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    How many hours of equivalent peak sun do you get per day?
    :param latitude: Latitude [degrees]
    :param day_of_year: Julian day (1 == Jan. 1, 365 == Dec. 31)
    :param time: Time since (local) solar noon [seconds]
    :return:
    """
    import warnings
    warnings.warn(
        "Use `solar_flux()` function from this module instead and integrate, which allows far greater granularity.",
        DeprecationWarning
    )
    time = np.linspace(0, 86400, 1000)
    fluxes = solar_flux(latitude, day_of_year, time)
    energy_per_area = np.sum(np.trapz(fluxes) * np.diff(time))

    duration_of_equivalent_peak_sun = energy_per_area / solar_flux(latitude, day_of_year, time=0.)

    sun_hours = duration_of_equivalent_peak_sun / 3600

    return sun_hours


def length_day(
        latitude: Union[float, np.ndarray],
        day_of_year: Union[int, float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Gives the duration where the sun is above the horizon on a given day.

    Args:

        latitude: Local geographic latitude [degrees]. Positive for north, negative for south.

        day_of_year: Julian day (1 == Jan. 1, 365 == Dec. 31)

    Returns: The duration where the sun is above the horizon on a given day. [seconds]

    """
    dec = declination_angle(day_of_year)

    constant = -np.sind(dec) * np.sind(latitude) / (np.cosd(dec) * np.cosd(latitude))

    sun_time_nondim = 2 * np.arccos(_prepare_for_inverse_trig(constant))
    sun_time = sun_time_nondim / (2 * np.pi) * 86400

    return sun_time


def mass_MPPT(
        power: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Gives the estimated mass of a Maximum Power Point Tracking (MPPT) unit for solar energy
    collection. Based on regressions at AeroSandbox/studies/SolarMPPTMasses.

    Args:
        power: Power of MPPT [watts]

    Returns:
        Estimated MPPT mass [kg]
    """
    constant = 0.066343
    exponent = 0.515140
    return constant * power ** exponent


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    # plt.switch_backend('WebAgg')

    base_color = p.palettes['categorical'][0]
    quality_colors = {
        'clean'   : p.adjust_lightness(base_color, amount=1.2),
        'typical' : p.adjust_lightness(base_color, amount=0.7),
        'polluted': p.adjust_lightness(base_color, amount=0.2),
    }

    ##### Plot solar_flux() over the course of a day
    time = np.linspace(0, 86400, 86401)
    hour = time / 3600

    base_kwargs = dict(
        latitude=23.5,
        day_of_year=172,
        time=time,
    )

    fig, ax = plt.subplots(2, 1, figsize=(7, 6.5))
    plt.sca(ax[0])
    plt.title(f"Solar Flux on a Horizontal Surface Over A Day\n(Tropic of Cancer, Summer Solstice, Sea Level)")
    for q in quality_colors.keys():
        plt.plot(
            hour,
            solar_flux(
                **base_kwargs,
                air_quality=q
            ),
            color=quality_colors[q],
            label=f'ASB Model: {q.capitalize()} air'
        )

    plt.sca(ax[1])
    plt.title(f"Solar Flux on a Sun-Tracking Surface Over A Day\n(Tropic of Cancer, Summer Solstice, Sea Level)")
    for q in quality_colors.keys():
        plt.plot(
            hour,
            solar_flux(
                **base_kwargs,
                panel_tilt_angle=90 - solar_elevation_angle(**base_kwargs),
                panel_azimuth_angle=solar_azimuth_angle(**base_kwargs),
                air_quality=q
            ),
            color=quality_colors[q],
            label=f'ASB Model: {q.capitalize()} air'
        )

    for a in ax:
        plt.sca(a)
        plt.xlabel("Time after Local Solar Noon [hours]")
        plt.ylabel("Solar Flux [$W/m^2$]")
        plt.xlim(0, 24)
        plt.ylim(-10, 1200)
        p.set_ticks(3, 0.5, 200, 50)

    plt.sca(ax[0])
    p.show_plot()

    ##### Plot solar_flux() as a function of elevation angle, and compare to data to validate.
    # Source: Ed. (2008), Table 1.1, Earthscan with the International Institute for Environment and Development, Deutsche Gesellschaft für Sonnenenergie. ISBN 1-84407-442-0.
    # Via: https://en.wikipedia.org/wiki/Air_mass_(solar_energy)#Solar_intensity

    # Values here give lower and upper bounds for measured solar flux on a typical clear day, varying primarily due
    # to pollution.
    raw_data = """\
z [deg],AM [-],Solar Flux Lower Bound [W/m^2],Solar Flux Upper Bound [W/m^2]
0,1,840,1130
23,1.09,800,1110
30,1.15,780,1100
45,1.41,710,1060
48.2,1.5,680,1050
60,2,560,970
70,2.9,430,880
75,3.8,330,800
80,5.6,200,660
85,10,85,480
90,38,6,34
"""
    import pandas as pd
    from io import StringIO

    delimiter = "\t"
    df = pd.read_csv(
        StringIO(raw_data),
        delimiter=','
    )
    df["Solar Flux [W/m^2]"] = (df['Solar Flux Lower Bound [W/m^2]'] + df['Solar Flux Upper Bound [W/m^2]']) / 2

    fluxes = solar_flux(
        **base_kwargs,
        panel_tilt_angle=90 - solar_elevation_angle(**base_kwargs),
        panel_azimuth_angle=solar_azimuth_angle(**base_kwargs),
    )
    elevations = solar_elevation_angle(
        **base_kwargs
    )

    fig, ax = plt.subplots()
    for q in quality_colors.keys():
        plt.plot(
            solar_elevation_angle(**base_kwargs),
            solar_flux(
                **base_kwargs,
                panel_tilt_angle=90 - solar_elevation_angle(**base_kwargs),
                panel_azimuth_angle=solar_azimuth_angle(**base_kwargs),
                air_quality=q
            ),
            color=quality_colors[q],
            label=f'ASB Model: {q.capitalize()} air',
            zorder=3
        )

    data_color = p.palettes['categorical'][1]

    plt.fill_between(
        x=90 - df['z [deg]'].values,
        y1=df['Solar Flux Lower Bound [W/m^2]'],
        y2=df['Solar Flux Upper Bound [W/m^2]'],
        color=data_color,
        alpha=0.4,
        label='Experimental Data Range\n(due to Pollution)',
        zorder=2.9,
    )
    for d in ['Lower', 'Upper']:
        plt.plot(
            90 - df['z [deg]'].values,
            df[f'Solar Flux {d} Bound [W/m^2]'],
            ".",
            color=data_color,
            alpha=0.7,
            zorder=2.95
        )

    plt.annotate(
        text='Data: "Planning and Installing Photovoltaic Systems".\nEarthscan (2008), ISBN 1-84407-442-0.',
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        ha="left",
        va='top',
        fontsize=9
    )
    plt.xlim(-5, 90)
    p.set_ticks(15, 5, 200, 50)

    p.show_plot(
        f"Sun Position vs. Solar Flux on a Sun-Tracking Surface",
        f"Solar Elevation Angle [deg]",
        "Solar Flux [$W/m^2$]"
    )
