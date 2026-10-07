import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)

datasets = [
    "Healthy Men",
    "First-Class Athletes",
    "World-Class Athletes",
]
for dataset in datasets:
    data = np.genfromtxt("data/" + dataset + ".csv", delimiter=",")
    data = data[data[:, 0].argsort()]
    plt.semilogx(
        data[:, 0],
        data[:, 1],
        ".",
        label=dataset,
    )

ax.text(
    0.01, 0.01,
    "Fig. 2.4, Bicycling Science by D. Wilson",
    fontsize=6,
    color="gray",
    ha="left",
    va="bottom",
    transform=ax.transAxes
)


p.show_plot("Maximum-Achievable Human Power Output on Bicycle", "Duration [mins]", "Maximum Sustainable Power [W]")

