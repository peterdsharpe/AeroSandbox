from aerosandbox.modeling.fitting import *
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=1)

datasets = [
    "Healthy Men",
    "First-Class Athletes",
    "World-Class Athletes",
]

fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
for dataset in datasets:
    data = np.genfromtxt("data/" + dataset + ".csv", delimiter=",")
    data = data[data[:, 0].argsort()]
    plt.semilogx(
        # plt.loglog(
        data[:, 0],
        data[:, 1],
        ".",
        label=dataset,
    )

plt.xlabel(r"Duration [mins]")
plt.ylabel(r"Maximum Sustainable Power [W]")
plt.title("Human Power Output\n(Fig. 2.4, Bicycling Science by D. Wilson)")
plt.tight_layout()
plt.legend()
plt.savefig("data.svg")
plt.show()
