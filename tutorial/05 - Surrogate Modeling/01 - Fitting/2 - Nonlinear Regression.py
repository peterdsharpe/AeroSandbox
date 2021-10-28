# %%

import aerosandbox as asb
import aerosandbox.numpy as np

from scipy import io
from pathlib import Path

root = Path(__file__).parent

# %%

data = io.loadmat(str(root / "data" / "wind_data_99.mat"))
lats_v = data["lats"].flatten()
alts_v = data["alts"].flatten()
speeds = data["speeds"].reshape(len(alts_v), len(lats_v)).T.flatten()

lats, alts = np.meshgrid(lats_v, alts_v, indexing="ij")
lats = lats.flatten()
alts = alts.flatten()

# %%

lats_scaled = (lats - 37.5) / 11.5
alts_scaled = (alts - 24200) / 24200
speeds_scaled = (speeds - 7) / 56

alt_diff = np.diff(alts_v)
alt_diff_aug = np.hstack((
    alt_diff[0],
    alt_diff,
    alt_diff[-1]
))
weights_1d = (alt_diff_aug[:-1] + alt_diff_aug[1:]) / 2
weights_1d = weights_1d / np.mean(weights_1d)
# region_of_interest = np.logical_and(
#     alts_v > 10000,
#     alts_v < 40000
# )
# true_weights = np.where(
#     region_of_interest,
#     2,
#     1
# )
weights = np.tile(weights_1d, (93, 1)).flatten()


# %%

def model(x, p):
    l = x["lats_scaled"]
    a = x["alts_scaled"]

    agc = p["agc"]
    agh = p["agh"]
    ags = p["ags"]
    aqc = p["aqc"]
    c0 = p["c0"]
    c12 = p["c12"]
    c21 = p["c21"]
    c4a = p["c4a"]
    c4c = p["c4c"]
    cg = p["cg"]
    cgc = p["cgc"]
    cqa = p["cqa"]
    cql = p["cql"]
    cqla = p["cqla"]
    lgc = p["lgc"]
    lgh = p["lgh"]
    lgs = p["lgs"]
    lqc = p["lqc"]

    return (
            c0  # Constant
            + cql * (l - lqc) ** 2  # Quadratic in latitude
            + cqa * (a - aqc) ** 2  # Quadratic in altitude
            + cqla * a * l  # Quadratic cross-term
            + cg * np.exp(-(  # Gaussian bump
            np.fabs(l - lgc) ** lgh / (2 * lgs ** 2) +  # Center/Spread in latitude
            np.fabs(a - agc) ** agh / (2 * ags ** 2) +  # Center/Spread in altitude
            cgc * a * l  # Gaussian cross-term
    ))
            + c4a * (a - c4c) ** 4  # Altitude quartic
            + c12 * l * a ** 2  # Altitude linear-quadratic
            + c21 * l ** 2 * a  # Latitude linear-quadratic
    )


fit = asb.FittedModel(
    model=model,
    x_data={
        "lats_scaled": lats_scaled,
        "alts_scaled": alts_scaled
    },
    y_data=speeds_scaled,
    parameter_guesses={
        "agc" : -0.5363,
        "agh" : 1.957,
        "ags" : 0.1459,
        "aqc" : -1.465,
        "c0"  : -0.517,
        "c12" : 0.08495,
        "c21" : -0.0252,
        "c4a" : 0.02259,
        "c4c" : 1.028,
        "cg"  : 0.8051,
        "cgc" : 0.2787,
        "cqa" : 0.1866,
        "cql" : 0.01651,
        "cqla": -0.1362,
        "lgc" : 0.6944,
        "lgh" : 2.078,
        "lgs" : 0.9806,
        "lqc" : 4.036,
    },
    # parameter_bounds={
    #     "agc" : (-0.5363, -0.5363),
    #     "agh" : (1.957, 1.957),
    #     "ags" : (0.1459, 0.1459),
    #     "aqc" : (-1.465, -1.465),
    #     "c0"  : (-0.517, -0.517),
    #     "c12" : (0.08495, 0.08495),
    #     "c21" : (-0.0252, -0.0252),
    #     "c4a" : (0.02259, 0.02259),
    #     "c4c" : (1.028, 1.028),
    #     "cg"  : (0.8051, 0.8051),
    #     "cgc" : (0.2787, 0.2787),
    #     "cqa" : (0.1866, 0.1866),
    #     "cql" : (0.01651, 0.01651),
    #     "cqla": (-0.1362, -0.1362),
    #     "lgc" : (0.6944, 0.6944),
    #     "lgh" : (2.078, 2.078),
    #     "lgs" : (0.9806, 0.9806),
    #     "lqc" : (4.036, 4.036),
    # },
    weights=weights,
    # put_residuals_in_logspace=True
)

# %%

from aerosandbox.tools.pretty_plots import plt, sns, mpl, show_plot

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(projection="3d")
ax.scatter(lats, alts / 1e3, speeds, marker=".", color="k", alpha=0.8, linewidth=0, label="Data")

lats_plot = np.linspace(lats.min(), lats.max(), 200)
alts_plot = np.linspace(alts.min(), alts.max(), 200)
Lats_plot, Alts_plot = np.meshgrid(lats_plot, alts_plot)

speeds_plot = fit({
    "lats_scaled": (Lats_plot.flatten() - 37.5) / 11.5,
    "alts_scaled": (Alts_plot.flatten() - 24200) / 24200,
}) * 56 + 7
Speeds_plot = speeds_plot.reshape(
    len(alts_plot),
    len(lats_plot),
)

ax.plot_surface(Lats_plot, Alts_plot / 1e3, Speeds_plot,
                cmap=plt.cm.viridis,
                edgecolors=(1, 1, 1, 0.5),
                linewidth=0.5,
                alpha=0.7,
                rcount=40,
                ccount=40,
                shade=True,
                )
ax.view_init(38, -130)
ax.set_xlabel("Latitude [deg. N]")
ax.set_ylabel("Altitude [km]")
ax.set_zlabel("Wind Speed [m/s]")
plt.title("99th-Percentile Wind Speeds\nContinental U.S., August, 1979-2020")
plt.legend()
plt.subplots_adjust(
    bottom=0.055,
    left=0,
    right=1,
    top=0.90
)
show_plot(tight_layout=False)
