import aerosandbox.numpy as np
from aerosandbox.atmosphere.atmosphere import Atmosphere
from typing import Union

"""
Welcome to the AeroSandbox solar energy library!

The function you're probably looking for is `solar_flux()`, which summarizes this entire module and computes the 
realized solar flux on a given surface as a function of many different parameters.
"""


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

    solar_elevation_angle = np.arcsind(
        np.sind(declination) * np.sind(latitude) +
        np.cosd(declination) * np.cosd(latitude) * np.cosd(time / 86400 * 360)
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
    cos_azimuth = np.clip(cos_azimuth, -1, 1)

    azimuth_raw = np.arccosd(cos_azimuth)

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
        altitude: Union[float, np.ndarray],
        panel_azimuth_angle: Union[float, np.ndarray] = 0.,
        panel_tilt_angle: Union[float, np.ndarray] = 0.,
        air_quality: str='typical',
        **deprecated_kwargs
) -> Union[float, np.ndarray]:
    """
    Computes the solar power flux (power per unit area) on a flat panel.

    Source for atmospheric absorption:

        * Planning and installing photovoltaic systems: a guide for installers, architects and engineers,
        2nd Ed. (2008), Table 1.1, Earthscan with the International Institute for Environment and Development,
        Deutsche Gesellschaft für Sonnenenergie. ISBN 1-84407-442-0., accessed via
        https://en.wikipedia.org/wiki/Air_mass_(solar_energy)

    Args:

        latitude: Local geographic latitude [degrees]. Positive for north, negative for south.

        day_of_year: Julian day (1 == Jan. 1, 365 == Dec. 31)

        time: Time after local solar noon [seconds]

        altitude: Altitude of the panel above sea level [meters]. This affects atmospheric absorption and scattering
        characteristics.

        panel_azimuth_angle: The azimuth angle of the panel normal [degrees] (the compass direction from which the
        sunlight is coming).

            * 0 corresponds to North, 90 corresponds to East.

            * Input ranges from 0 to 360 degrees.

        panel_tilt_angle: The angle between the panel normal and vertical (zenith) [degrees].

            * Note: this angle convention is different than the solar elevation angle convention!

            * A horizontal panel has a tilt angle of 0, and a vertical panel has a tilt angle of 90 degrees.

        air_quality: Indicates the amount of pollution in the air. A string, one of:

            * 'typical': Corresponds to "rural aerosol loading" following ASTM G-173.

            * 'clean': Pristine atmosphere conditions.

            * 'polluted': Urban atmosphere conditions.

    Returns: The solar power flux [W/m^2].

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

    diffuse_irradiance = scattering_losses * atmospheric_transmission_fraction
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
    times = np.linspace(0, 86400, 1000)
    dt = np.diff(times)
    normalized_fluxes = (
        # solar_flux_outside_atmosphere_normal(day_of_year) *
        incidence_angle_function(latitude, day_of_year, times, scattering)
    )
    sun_hours = np.sum(
        (normalized_fluxes[1:] + normalized_fluxes[:-1]) / 2 * dt
    ) / 3600

    return sun_hours


def length_day(
        latitude: Union[float, np.ndarray],
        day_of_year: Union[int, float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Gives the duration where the sun is above the horizon on a given day.

    Args:
        latitude:
        day_of_year:

    Returns:

    """
    """
    Gives the duration where the sun is above the horizon on a given day.

    :param latitude: Latitude [degrees]
    :param day_of_year: Julian day (1 == Jan. 1, 365 == Dec. 31)
    :return: Seconds of sunlight in a given day
    """
    dec = declination_angle(day_of_year)

    constant = -np.sind(dec) * np.sind(latitude) / (np.cosd(dec) * np.cosd(latitude))
    constant = np.clip(constant, -1, 1)

    sun_time_nondim = 2 * np.arccos(constant)
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

    time = np.linspace(0, 86400, 86401)
    hour = time / 3600

    fig, ax = plt.subplots(figsize=(5, 3))
    fluxes = lambda **kwargs: solar_flux(
        latitude=23.5,
        day_of_year=172,
        time=time,
        altitude=0,
        **kwargs
    )
    for q in ['clean', 'typical', 'polluted']:
        plt.plot(hour, fluxes(air_quality=q), label=f'{q.capitalize()} air')

    p.set_ticks(3, 0.5)
    plt.xlim(0, 24)
    p.show_plot(
        f"Solar Flux on a Horizontal Surface Over A Day\n(Tropic of Cancer, Summer Solstice)",
        "Time after Local Solar Noon [hours]",
        "Solar Flux [$W/m^2$]"
    )

    fig, ax = plt.subplots(figsize=(5, 3))
    fluxes = lambda **kwargs: solar_flux(
        latitude=23.5,
        day_of_year=172,
        time=time,
        altitude=0,
        panel_tilt_angle=90 - solar_elevation_angle(23.5, 177, time),
        panel_azimuth_angle = solar_azimuth_angle(23.5, 177, time),
        **kwargs
    )
    for q in ['clean', 'typical', 'polluted']:
        plt.plot(hour, fluxes(air_quality=q), label=f'{q.capitalize()} air')

    p.set_ticks(3, 0.5)
    plt.xlim(0, 24)
    p.show_plot(
        f"Solar Flux on a Sun-Tracking Surface Over A Day\n(Tropic of Cancer, Summer Solstice)",
        "Time after Local Solar Noon [hours]",
        "Solar Flux [$W/m^2$]"
    )
