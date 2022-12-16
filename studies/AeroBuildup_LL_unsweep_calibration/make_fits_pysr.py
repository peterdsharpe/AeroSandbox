import aerosandbox as asb
import aerosandbox.numpy as np
import pandas as pd
from pysr import PySRRegressor

df = pd.read_csv("data.csv")

x_data = {
    "a": (df["AR"] / (df["AR"] + 2)).values,
    "t": np.exp(-df["taper"].values),
    "s": np.radians(df["sweep"].values),
    # "sin_sweep": np.sind(df["sweep"]).values,
    # "cos_sweep": np.cosd(df["sweep"]).values,
}
y_data = ((df["ab_xnp"] - df["vlm_xnp"]) / (df["MAC"])).values

bad_data = np.abs(y_data) > 1
x_data = {
    k: v[~bad_data]
    for k, v in x_data.items()
}
y_data = y_data[~bad_data]

# x_data["cos_sweep"] = np.cosd(x_data["sweep"])

X = np.vstack(tuple(x_data.values())).T

y = y_data

model = PySRRegressor(
    niterations=1000000,  # < Increase me for better results
    population_size=50,
    ncyclesperiteration=700,
    binary_operators=[
        "*",
        "+",
        "pow"
    ],
    unary_operators=[
        # "cos",
        # "exp",
        # "sin",
        # "tan",
        # "inv(x) = 1/x",
        # ^ Custom operator (julia syntax)
    ],
    complexity_of_operators={
        "*"  : 1,
        "+"  : 1,
        "pow": 1,
        # "exp": 2,
        # "cos": 3,
        # "sin": 3,
        # "tan": 5,
    },
    # complexity_of_constants=0.5,
    complexity_of_variables=2,
    constraints={
        'pow': (-1, 5),
        # 'sin': 5,
        # 'cos': 5,
        # 'tan': 5,
    },
    maxsize=30,
    # batching=True,
    # batch_size=500,
    # warm_start=True,
    # extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    loss="loss(prediction, target, weight) = weight * abs(prediction - target)",
    # ^ Custom loss function (julia syntax)
)

weights = np.ones_like(df["AR"].values, dtype="float32")
weights[df["AR"].values > 1] *= 2
weights[df["taper"] < 1] *= 4
weights[df["taper"] > 2] *= 0.25
weights[df["sweep"].values >= 0] *= 6
weights[np.abs(df["sweep"].values) < 16] *= 2
weights[np.abs(df["sweep"].values) < 31] *= 2

weights = weights[~bad_data]
weights /= np.mean(weights)  # Renormalize

model.fit(
    X, y,
    weights=weights,
    variable_names=list(x_data.keys())
)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    fig, ax = plt.subplots()
    plt.plot(
        np.degrees(x_data["sweep_rad"]),
        y_data,
        ".k",
        alpha=0.2
    )
    sweep_plot = np.linspace(-90, 90, 500)

    sweep_rad = np.radians(sweep_plot)
    AR = 1e-6
    ARf = AR / (AR + 2)
    taper = 1

    # dev = ((((sweep_rad - taper) * -0.30672163) * sweep_rad) * np.tan(ARf))
    dev = (((((sweep_rad * ARf) - (taper - 0.29707846)) * sweep_rad) * (np.tan(ARf) * -0.35278562)) + (
            0.017927283 / ARf))

    plt.plot(
        sweep_plot,
        dev
    )
    plt.ylim(-1, 0.5)
    p.show_plot()
