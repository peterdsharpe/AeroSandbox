import aerosandbox.numpy as np


# ##### Winds
# # Fixed value
# wind_speed = 0  # m/s (19.5 m/s is 99% wind speed @ 60000 ft)
# wind_speed_midpoints = wind_speed
#
# ## 2D differentiable interpolant
# winds_altitudes = np.load("altitudes.npy")
# winds_latitudes = np.load("latitudes.npy")
# winds_speeds = np.load("wind_99_vs_latitudes_altitudes.npy")
#
# winds_altitudes = winds_altitudes[::-1]
# winds_latitudes = winds_latitudes[::-1]
# winds_speeds = winds_speeds[::-1, ::-1].ravel(order="F")
#
# wind_speed_func = cas.interpolant('name', 'linear', [winds_altitudes, winds_latitudes], winds_speeds.ravel(order="F"))
#
# wind_speed_inputs = cas.transpose(cas.horzcat(
#     y,
#     latitude * cas.GenDM_ones(n_timesteps)
# ))
# wind_speed_inputs_trapz = cas.transpose(cas.horzcat(
#     trapz(y),
#     latitude * cas.GenDM_ones(n_timesteps - 1)
# ))
# assert latitude >= winds_latitudes[0]
# assert latitude <= winds_latitudes[-1]
# wind_speed = cas.transpose(wind_speed_func(wind_speed_inputs))
# wind_speed_midpoints = cas.transpose(wind_speed_func(wind_speed_inputs_trapz))

## Curve fit
def wind_speed_conus_summer_99(altitude, latitude):
    """
    Returns the 99th-percentile wind speed magnitude over the continental United States (CONUS) in July-Aug. Aggregate of data from 1972 to 2019.
    Fits at C:\Projects\GitHub\Wind_Analysis
    :param altitude: altitude [m]
    :param latitude: latitude [deg]
    :return: 99th-percentile wind speed over the continental United States in the summertime. [m/s]
    """
    l = (latitude - 37.5) / 11.5
    a = (altitude - 24200) / 24200

    agc = -0.5363486000267786
    agh = 1.9569754777072828
    ags = 0.1458701999734713
    aqc = -1.4645014948089652
    c0 = -0.5169694086686631
    c12 = 0.0849519807021402
    c21 = -0.0252010113059998
    c4a = 0.0225856848053377
    c4c = 1.0281877353734501
    cg = 0.8050736230004489
    cgc = 0.2786691793571486
    cqa = 0.1866078047914259
    cql = 0.0165126852561671
    cqla = -0.1361667658248024
    lgc = 0.6943655538727291
    lgh = 2.0777449841036777
    lgs = 0.9805766577269118
    lqc = 4.0356834595743214

    s = c0 + cql * (l - lqc) ** 2 + cqa * (a - aqc) ** 2 + cqla * a * l + cg * np.exp(
        -(np.fabs(l - lgc) ** lgh / (2 * lgs ** 2) + np.fabs(a - agc) ** agh / (
                    2 * ags ** 2) + cgc * a * l)) + c4a * (
                a - c4c) ** 4 + c12 * l * a ** 2 + c21 * l ** 2 * a

    speed = s * 56 + 7
    return speed
