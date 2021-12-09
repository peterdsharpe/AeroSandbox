import aerosandbox.numpy as np


def solar_flux_outside_atmosphere_normal(day_of_year):
    """
    Normal solar flux at the top of the atmosphere (variation due to orbital eccentricity)
    :param day_of_year: Julian day (1 == Jan. 1, 365 == Dec. 31)
    :return: Solar flux [W/m^2]
    """
    # Space effects
    # # Source: https://www.itacanet.org/the-sun-as-a-source-of-energy/part-2-solar-energy-reaching-the-earths-surface/#2.1.-The-Solar-Constant
    # solar_flux_outside_atmosphere_normal = 1367 * (1 + 0.034 * cas.cos(2 * cas.pi * (day_of_year / 365.25)))  # W/m^2; variation due to orbital eccentricity
    # Source: https://www.pveducation.org/pvcdrom/properties-of-sunlight/solar-radiation-outside-the-earths-atmosphere
    return 1366 * (
            1 + 0.033 * np.cosd(360 * (day_of_year - 2) / 365))  # W/m^2; variation due to orbital eccentricity


def declination_angle(day_of_year):
    """
    Declination angle, in degrees, as a func. of day of year. (Seasonality)
    :param day_of_year: Julian day (1 == Jan. 1, 365 == Dec. 31)
    :return: Declination angle [deg]
    """
    # Declination (seasonality)
    # Source: https://www.pveducation.org/pvcdrom/properties-of-sunlight/declination-angle
    return -23.45 * np.cosd(360 / 365 * (day_of_year + 10))  # in degrees


def solar_elevation_angle(latitude, day_of_year, time):
    """
    Elevation angle of the sun [degrees] for a local observer.

    :param latitude: Latitude [degrees]

    :param day_of_year: Julian day (1 == Jan. 1, 365 == Dec. 31)

    :param time: Time after local solar noon [seconds]

    :return: Solar elevation angle [degrees] (angle between horizon and sun). Returns negative values if the sun is
    below the horizon.
    """

    # Solar elevation angle (including seasonality, latitude, and time of day)
    # Source: https://www.pveducation.org/pvcdrom/properties-of-sunlight/elevation-angle
    declination = declination_angle(day_of_year)

    solar_elevation_angle = np.arcsind(
        np.sind(declination) * np.sind(latitude) +
        np.cosd(declination) * np.cosd(latitude) * np.cosd(time / 86400 * 360)
    )  # in degrees
    return solar_elevation_angle


def solar_azimuth_angle(latitude, day_of_year, time):
    """
    Zenith angle of the sun [degrees] for a local observer.
    :param latitude: Latitude [degrees]
    :param day_of_year: Julian day (1 == Jan. 1, 365 == Dec. 31)
    :param time: Time after local solar noon [seconds]
    :return: Solar azimuth angle [degrees] (the compass direction from which the sunlight is coming).
    """

    # Solar azimuth angle (including seasonality, latitude, and time of day)
    # Source: https://www.pveducation.org/pvcdrom/properties-of-sunlight/azimuth-angle
    declination = declination_angle(day_of_year)
    sdec = np.sind(declination)
    cdec = np.cosd(declination)
    slat = np.sind(latitude)
    clat = np.cosd(latitude)
    ctime = np.cosd(time / 86400 * 360)

    elevation = solar_elevation_angle(latitude, day_of_year, time)
    cele = np.cosd(elevation)

    azimuth_raw = np.arccosd(
        (sdec * clat - cdec * slat * ctime) / cele
    )

    is_solar_morning = np.mod(time, 86400) > 43200

    solar_azimuth_angle = np.where(
        is_solar_morning,
        azimuth_raw,
        360 - azimuth_raw
    )

    return solar_azimuth_angle


def incidence_angle_function(
        latitude: float,
        day_of_year: float,
        time: float,
        panel_azimuth_angle: float = 0,
        panel_tilt_angle: float = 0,
        scattering: bool = True,
):
    """
    This website will be useful for accounting for direction of the vertical surface
    https://www.pveducation.org/pvcdrom/properties-of-sunlight/arbitrary-orientation-and-tilt
    :param latitude: Latitude [degrees]
    :param day_of_year: Julian day (1 == Jan. 1, 365 == Dec. 31)
    :param time: Time since (local) solar noon [seconds]
    :param panel_azimuth_angle: the directionality of the solar panel (0 degrees if pointing North and 90 if East)
    :param panel_tilt_angle: the degrees of horizontal the array is mounted (0 if horizontal and 90 if vertical)
    :param scattering: Boolean: include scattering effects at very low angles?

    :returns
    illumination_factor: Fraction of solar insolation received, relative to what it would get if it were perfectly oriented to the sun.
    """
    solar_elevation = solar_elevation_angle(latitude, day_of_year, time)
    solar_azimuth = solar_azimuth_angle(latitude, day_of_year, time)
    cosine_factor = (
            np.cosd(solar_elevation) *
            np.sind(panel_tilt_angle) *
            np.cosd(panel_azimuth_angle - solar_azimuth)
            + np.sind(solar_elevation) * np.cosd(panel_tilt_angle)
    )
    if scattering:
        illumination_factor = cosine_factor * scattering_factor(solar_elevation)
    else:
        illumination_factor = cosine_factor

    illumination_factor = np.fmax(illumination_factor, 0)
    illumination_factor = np.where(
        solar_elevation < 0,
        0,
        illumination_factor
    )
    return illumination_factor


def scattering_factor(elevation_angle):
    """
    Calculates a scattering factor (a factor that gives losses due to atmospheric scattering at low elevation angles).
    Source: AeroSandbox/studies/SolarPanelScattering
    :param elevation_angle: Angle between the horizon and the sun [degrees]
    :return: Fraction of the light that is not lost to scattering.
    # TODO figure out if scattering is w.r.t. sun elevation or w.r.t. angle between panel and sun
    """
    elevation_angle = np.clip(elevation_angle, 0, 90)
    theta = 90 - elevation_angle  # Angle between panel normal and the sun, in degrees

    # # Model 1
    # c = (
    #     0.27891510500505767300438719757949,
    #     -0.015994330894744987481281839336589,
    #     -19.707332432605799255043166340329,
    #     -0.66260979582573353852126274432521
    # )
    # factor = c[0] + c[3] * theta_rad + cas.exp(
    #     c[1] * (
    #             cas.tan(theta_rad) + c[2] * theta_rad
    #     )
    # )

    # Model 2
    c = (
        -0.04636,
        -0.3171
    )
    factor = np.exp(
        c[0] * (
                np.tand(theta * 0.999) + c[1] * np.radians(theta)
        )
    )

    # # Model 3
    # p1 = -21.74
    # p2 = 282.6
    # p3 = -1538
    # p4 = 1786
    # q1 = -923.2
    # q2 = 1456
    # x = theta_rad
    # factor = ((p1*x**3 + p2*x**2 + p3*x + p4) /
    #            (x**2 + q1*x + q2))

    # Keep this:
    # factor = cas.fmin(cas.fmax(scattering_factor, 0), 1)
    return factor


def solar_flux(
        latitude: float,
        day_of_year: float,
        time: float,
        panel_azimuth_angle: float = 0,
        panel_tilt_angle: float = 0,
        scattering: bool = True
) -> float:
    """
    What is the solar flux on a horizontal surface for some given conditions?
    :param latitude: Latitude [degrees]
    :param day_of_year: Julian day (1 == Jan. 1, 365 == Dec. 31)
    :param time: Time since (local) solar noon [seconds]
    :param scattering: Boolean: include scattering effects at very low angles?
    :return:
    """
    return (
            solar_flux_outside_atmosphere_normal(day_of_year) *
            incidence_angle_function(
                latitude=latitude,
                day_of_year=day_of_year,
                time=time,
                panel_azimuth_angle=panel_azimuth_angle,
                panel_tilt_angle=panel_tilt_angle,
                scattering=scattering
            )
    )


def solar_flux_circular_flight_path(
        latitude: float,
        day_of_year: float,
        time: float,
        panel_angle: float,
        panel_heading: float,
        scattering: bool = True
) -> float:
    """
    What is the solar flux on a surface at a given angle for a circular flight path given the radius?
    :param latitude: Latitude [degrees]
    :param day_of_year: Julian day (1 == Jan. 1, 365 == Dec. 31)
    :param time: Time since (local) solar noon [seconds]
    :param panel_angle: the degrees of horizontal the array is mounted (0 if hoirzontal and 90 if vertical)
    :param panel_heading: the directionality of the solar panel (0 degrees if pointing North and 180 if South)
    :param scattering: Boolean: include scattering effects at very low angles?
    :return:
    """

    solar_flux_on_panel = solar_flux_outside_atmosphere_normal(day_of_year) * incidence_angle_function(
        latitude, day_of_year, time, panel_heading, panel_angle)
    return solar_flux_on_panel


def peak_sun_hours_per_day_on_horizontal(latitude, day_of_year, scattering=True):
    """
    How many hours of equivalent peak sun do you get per day?
    :param latitude: Latitude [degrees]
    :param day_of_year: Julian day (1 == Jan. 1, 365 == Dec. 31)
    :param time: Time since (local) solar noon [seconds]
    :param scattering: Boolean: include scattering effects at very low angles?
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


def length_day(latitude, day_of_year):
    """
    How many hours of with incoming solar energy per day?

    Warning: NOT differentiable as-written # TODO make differentiable
    :param latitude: Latitude [degrees]
    :param day_of_year: Julian day (1 == Jan. 1, 365 == Dec. 31)
    :return: hours of no sun
    """
    times = np.linspace(0, 86400, 100)
    dt = np.diff(times)
    sun_time = 0
    for time in times:
        current_solar_flux = solar_flux(
            latitude, day_of_year, time, scattering=True
        )
        if current_solar_flux > 1:
            sun_time = sun_time + (872.72727273 / 60 / 60)
    return sun_time


def mass_MPPT(
        power: float
) -> float:
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
    np.seterr(all='raise')
    incidence_angle_function(latitude=0, day_of_year=80, time=0)
    pass
