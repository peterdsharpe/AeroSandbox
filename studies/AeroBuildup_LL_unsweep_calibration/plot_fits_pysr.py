import aerosandbox.numpy as np
import pandas as pd
from scipy import interpolate

### Clean up PySR model

pysr_model = """
((((((((s * a) + ((t * 2.0672152) + -1.5531529)) * a) * s) + (0.39056087 ^ a)) * a) + -0.39042628) * -0.52582735)
"""

import sympy as sym

a, s, t = sym.symbols('a s t')
pysr_model_sympy = eval(pysr_model.replace("^", "**").replace("\n", "")).simplify()
pysr_model_lambda = sym.lambdify([a, s, t], pysr_model_sympy)

print(f"Simplifed Model\n{pysr_model_sympy}")

### Get data
df = pd.read_csv("data.csv")

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    fig, ax = plt.subplots()
    plt.plot(
        df["sweep"],
        ((df["ab_xnp"] - df["vlm_xnp"]) / (df["MAC"])).values,
        ".k",
        alpha=0.2
    )
    sweep_plot = np.linspace(-90, 90, 500)

    AR = 10
    sweep_rad = np.radians(sweep_plot)
    taper = 0.5

    a = AR / (AR + 2)
    s = sweep_rad
    t = np.exp(-taper)

    dev = pysr_model_lambda(a, s, t)

    interpolator = interpolate.LinearNDInterpolator(
        np.stack([df["AR"], df["sweep"], df["taper"]], axis=1),
        ((df["ab_xnp"] - df["vlm_xnp"]) / (df["MAC"])).values,
        rescale=True
    )

    plt.plot(
        sweep_plot,
        interpolator(
            np.stack([
                AR * np.ones_like(sweep_plot),
                sweep_plot,
                taper * np.ones_like(sweep_plot)
            ], axis=1)
        ),
        label="Data"
    )

    plt.plot(
        sweep_plot,
        dev,
        label="Model"
    )
    plt.ylim(-0.5, 0.3)
    p.show_plot()
