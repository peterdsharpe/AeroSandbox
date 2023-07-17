import aerosandbox.numpy as np
import datetime
from aerosandbox.modeling.interpolation import InterpolatedModel
from pathlib import Path
import os


def wind_speed_conus_summer_99(altitude, latitude):
    """
    Returns the 99th-percentile wind speed magnitude over the continental United States (CONUS) in July-Aug. Aggregate of data from 1972 to 2019.
    Fits at C:\Projects\GitHub\Wind_Analysis
    :param altitude: altitude [m]
    :param latitude: latitude, in degrees North [deg]
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


### Prep data for global wind speed function
# Import data
root = Path(os.path.abspath(__file__)).parent
altitudes_world = np.load(root / "datasets" / "winds_and_tropopause_global" / "altitudes.npy")
latitudes_world = np.load(root / "datasets" / "winds_and_tropopause_global" / "latitudes.npy")
day_of_year_world_boundaries = np.linspace(0, 365, 13)
day_of_year_world = (day_of_year_world_boundaries[1:] + day_of_year_world_boundaries[:-1]) / 2
winds_95_world = np.load(root / "datasets" / "winds_and_tropopause_global" / "winds_95_vs_altitude_latitude_day.npy")

# Trim the poles
latitudes_world = latitudes_world[1:-1]
winds_95_world = winds_95_world[:, 1:-1, :]

# Flip data appropriately
altitudes_world = np.flip(altitudes_world)
latitudes_world = np.flip(latitudes_world)
### NOTE: winds_95_world has *already* been flipped appropriately

# # Extend altitude range down to the ground # TODO review and redo properly
# altitudes_world_to_extend = [-1000, 0, 5000]
# altitudes_world = np.hstack((
#     altitudes_world_to_extend,
#     altitudes_world
# ))
# winds_95_world = np.concatenate(
#     (
#         np.tile(
#             winds_95_world[0, :, :],
#             (3, 1, 1)
#         ),
#         winds_95_world
#     ),
#     axis=0
# )

# Downsample
latitudes_world = latitudes_world[::5]
winds_95_world = winds_95_world[:, ::5, :]

# Extend boundaries so that cubic spline interpolates around day_of_year appropriately.
extend_bounds = 3
day_of_year_world = np.hstack((
    day_of_year_world[-extend_bounds:] - 365,
    day_of_year_world,
    day_of_year_world[:extend_bounds] + 365
))
winds_95_world = np.dstack((
    winds_95_world[:, :, -extend_bounds:],
    winds_95_world,
    winds_95_world[:, :, :extend_bounds]
))

# Make the model
winds_95_world_model = InterpolatedModel(
    x_data_coordinates={
        "altitude"   : altitudes_world,
        "latitude"   : latitudes_world,
        "day of year": day_of_year_world,
    },
    y_data_structured=winds_95_world,
)


def wind_speed_world_95(
        altitude,
        latitude,
        day_of_year
):
    """
    Gives the 95th-percentile wind speed as a function of altitude, latitude, and day of year.
    Args:
        altitude: Altitude, in meters
        latitude: Latitude, in degrees north
        day_of_year: Day of year (Julian day), in range 0 to 365

    Returns: The 95th-percentile wind speed, in meters per second.

    """

    return winds_95_world_model({
        "altitude"   : altitude,
        "latitude"   : latitude,
        "day of year": day_of_year
    })


### Prep data for tropopause altitude function
# Import data
latitudes_trop = np.linspace(-80, 80, 50)
day_of_year_trop_boundaries = np.linspace(0, 365, 13)
day_of_year_trop = (day_of_year_trop_boundaries[1:] + day_of_year_trop_boundaries[:-1]) / 2
tropopause_altitude_km = np.genfromtxt(
    root / "datasets" / "winds_and_tropopause_global" / "strat-height-monthly.csv",
    delimiter=","
)

# Extend boundaries
extend_bounds = 3
day_of_year_trop = np.hstack((
    day_of_year_trop[-extend_bounds:] - 365,
    day_of_year_trop,
    day_of_year_trop[:extend_bounds] + 365
))
tropopause_altitude_km = np.hstack((
    tropopause_altitude_km[:, -extend_bounds:],
    tropopause_altitude_km,
    tropopause_altitude_km[:, :extend_bounds]
))

# Make the model
tropopause_altitude_model = InterpolatedModel(
    x_data_coordinates={
        "latitude"   : latitudes_trop,
        "day of year": day_of_year_trop
    },
    y_data_structured=tropopause_altitude_km * 1e3
)


def tropopause_altitude(
        latitude,
        day_of_year
):
    """
    Gives the altitude of the tropopause (as determined by the altitude where lapse rate >= 2 C/km) as a function of
    latitude and day of year.

    Args:
        latitude: Latitude, in degrees north
        day_of_year: Day of year (Julian day), in range 0 to 365

    Returns: The tropopause altitude, in meters.

    """
    return tropopause_altitude_model({
        "latitude"   : latitude,
        "day of year": day_of_year
    })


if __name__ == '__main__':

    from aerosandbox.tools.pretty_plots import plt, sns, mpl, show_plot


    def plot_winds_at_altitude(altitude=18000):
        fig, ax = plt.subplots()

        day_of_years = np.linspace(0, 365, 150)
        latitudes = np.linspace(-80, 80, 120)
        Day_of_years, Latitudes = np.meshgrid(day_of_years, latitudes)

        winds = wind_speed_world_95(
            altitude=altitude * np.ones_like(Latitudes.flatten()),
            latitude=Latitudes.flatten(),
            day_of_year=Day_of_years.flatten(),
        ).reshape(Latitudes.shape)

        args = [
            day_of_years,
            latitudes,
            winds
        ]

        levels = np.arange(0, 80.1, 5)
        CS = plt.contour(*args, levels=levels, linewidths=0.5, colors="k", alpha=0.7)
        CF = plt.contourf(*args, levels=levels, cmap='viridis_r', alpha=0.7, extend="max")
        cbar = plt.colorbar(label="Wind Speed [m/s]", extendrect=True)
        ax.clabel(CS, inline=1, fontsize=9, fmt="%.0f m/s")

        plt.xticks(
            np.linspace(0, 365, 13)[:-1],
            (
                "Jan. 1",
                "Feb. 1",
                "Mar. 1",
                "Apr. 1",
                "May 1",
                "June 1",
                "July 1",
                "Aug. 1",
                "Sep. 1",
                "Oct. 1",
                "Nov. 1",
                "Dec. 1"
            ),
            rotation=40
        )

        lat_label_vals = np.arange(-80, 80.1, 20)
        lat_labels = []
        for lat in lat_label_vals:
            if lat >= 0:
                lat_labels.append(f"{lat:.0f}N")
            else:
                lat_labels.append(f"{-lat:.0f}S")
        plt.yticks(
            lat_label_vals,
            lat_labels
        )

        show_plot(
            f"95th-Percentile Wind Speeds at {altitude / 1e3:.0f} km Altitude",
            xlabel="Day of Year",
            ylabel="Latitude",
        )


    def plot_winds_at_day(day_of_year=0):
        fig, ax = plt.subplots()

        altitudes = np.linspace(0, 30000, 150)
        latitudes = np.linspace(-80, 80, 120)
        Altitudes, Latitudes = np.meshgrid(altitudes, latitudes)

        winds = wind_speed_world_95(
            altitude=Altitudes.flatten(),
            latitude=Latitudes.flatten(),
            day_of_year=day_of_year * np.ones_like(Altitudes.flatten()),
        ).reshape(Altitudes.shape)

        args = [
            altitudes / 1e3,
            latitudes,
            winds
        ]

        levels = np.arange(0, 80.1, 5)
        CS = plt.contour(*args, levels=levels, linewidths=0.5, colors="k", alpha=0.7)
        CF = plt.contourf(*args, levels=levels, cmap='viridis_r', alpha=0.7, extend="max")
        cbar = plt.colorbar(label="Wind Speed [m/s]", extendrect=True)
        ax.clabel(CS, inline=1, fontsize=9, fmt="%.0f m/s")

        lat_label_vals = np.arange(-80, 80.1, 20)
        lat_labels = []
        for lat in lat_label_vals:
            if lat >= 0:
                lat_labels.append(f"{lat:.0f}N")
            else:
                lat_labels.append(f"{-lat:.0f}S")
        plt.yticks(
            lat_label_vals,
            lat_labels
        )

        show_plot(
            f"95th-Percentile Wind Speeds at Day {day_of_year:.0f}",
            xlabel="Altitude [km]",
            ylabel="Latitude",
        )


    def plot_tropopause_altitude():
        fig, ax = plt.subplots()

        day_of_years = np.linspace(0, 365, 250)
        latitudes = np.linspace(-80, 80, 200)
        Day_of_years, Latitudes = np.meshgrid(day_of_years, latitudes)

        trop_alt = tropopause_altitude(
            Latitudes.flatten(),
            Day_of_years.flatten()
        ).reshape(Latitudes.shape)

        args = [
            day_of_years,
            latitudes,
            trop_alt / 1e3
        ]

        levels = np.arange(10, 20.1, 1)
        CS = plt.contour(*args, levels=levels, linewidths=0.5, colors="k", alpha=0.7)
        CF = plt.contourf(*args, levels=levels, cmap='viridis_r', alpha=0.7, extend="both")
        cbar = plt.colorbar(label="Tropopause Altitude [km]", extendrect=True)
        ax.clabel(CS, inline=1, fontsize=9, fmt="%.0f km")

        plt.xticks(
            np.linspace(0, 365, 13)[:-1],
            (
                "Jan. 1",
                "Feb. 1",
                "Mar. 1",
                "Apr. 1",
                "May 1",
                "June 1",
                "July 1",
                "Aug. 1",
                "Sep. 1",
                "Oct. 1",
                "Nov. 1",
                "Dec. 1"
            ),
            rotation=40
        )

        lat_label_vals = np.arange(-80, 80.1, 20)
        lat_labels = []
        for lat in lat_label_vals:
            if lat >= 0:
                lat_labels.append(f"{lat:.0f}N")
            else:
                lat_labels.append(f"{-lat:.0f}S")
        plt.yticks(
            lat_label_vals,
            lat_labels
        )

        show_plot(
            f"Tropopause Altitude by Season and Latitude",
            xlabel="Day of Year",
            ylabel="Latitude",
        )


    def plot_winds_at_tropopause_altitude():
        fig, ax = plt.subplots()

        day_of_years = np.linspace(0, 365, 150)
        latitudes = np.linspace(-80, 80, 120)
        Day_of_years, Latitudes = np.meshgrid(day_of_years, latitudes)

        winds = wind_speed_world_95(
            altitude=tropopause_altitude(Latitudes.flatten(), Day_of_years.flatten()),
            latitude=Latitudes.flatten(),
            day_of_year=Day_of_years.flatten(),
        ).reshape(Latitudes.shape)

        args = [
            day_of_years,
            latitudes,
            winds
        ]

        levels = np.arange(0, 80.1, 5)
        CS = plt.contour(*args, levels=levels, linewidths=0.5, colors="k", alpha=0.7)
        CF = plt.contourf(*args, levels=levels, cmap='viridis_r', alpha=0.7, extend="max")
        cbar = plt.colorbar(label="Wind Speed [m/s]", extendrect=True)
        ax.clabel(CS, inline=1, fontsize=9, fmt="%.0f m/s")

        plt.xticks(
            np.linspace(0, 365, 13)[:-1],
            (
                "Jan. 1",
                "Feb. 1",
                "Mar. 1",
                "Apr. 1",
                "May 1",
                "June 1",
                "July 1",
                "Aug. 1",
                "Sep. 1",
                "Oct. 1",
                "Nov. 1",
                "Dec. 1"
            ),
            rotation=40
        )

        lat_label_vals = np.arange(-80, 80.1, 20)
        lat_labels = []
        for lat in lat_label_vals:
            if lat >= 0:
                lat_labels.append(f"{lat:.0f}N")
            else:
                lat_labels.append(f"{-lat:.0f}S")
        plt.yticks(
            lat_label_vals,
            lat_labels
        )

        show_plot(
            f"95th-Percentile Wind Speeds at Tropopause Altitude",
            xlabel="Day of Year",
            ylabel="Latitude",
        )


    # plot_winds_at_altitude(altitude=18000)
    # plot_winds_at_day(day_of_year=0)
    # plot_tropopause_altitude()
    plot_winds_at_tropopause_altitude()
