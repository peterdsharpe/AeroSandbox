from aerosandbox.tools.fitting import *
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

sns.set(font_scale=1)

datasets = [
    "Healthy Men",
    "First-Class Athletes",
    "World-Class Athletes",
]

for dataset in datasets:
    # Ingest data
    data = np.genfromtxt(
        "data/" + dataset + ".csv",
        delimiter=","
    )
    data = data[data[:, 0].argsort()]
    durations = data[:, 0]  # in minutes
    powers = data[:, 1]  # in Watts


    def human_power_model(x, p):
        d = x["d"]
        logd = cas.log10(d)

        return (
                p["a"] * d ** (
                p["b0"] + p["b1"] * logd + p["b2"] * logd ** 2
        )
        ) # essentially, a cubic in log-log space


    params = fit(
        model=human_power_model,
        x_data={"d": durations},
        y_data=powers,
        param_guesses={
            "a" : 408,
            "b0": -0.17,
            "b1": 0.08,
            "b2": -0.04,
        },
        put_residuals_in_logspace=True
    )

    # Plot fit
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
    plt.loglog(durations, powers, ".")
    durations = np.logspace(-2, 4)
    plt.loglog(durations, human_power_model({"d": durations}, params))
    plt.xlabel(r"Duration [mins]")
    plt.ylabel(r"Maximum Sustainable Power [W]")
    plt.title("Human Power Output (%s)\n(Fig. 2.4, Bicycling Science by D. Wilson)" % dataset)
    plt.tight_layout()
    plt.show()
