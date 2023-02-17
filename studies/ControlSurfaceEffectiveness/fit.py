from pysr import PySRRegressor
from read_data import a, d, hf, lr, eff
import aerosandbox as asb
import aerosandbox.numpy as np


def model(x, p):
    return 1 - (1 - x + 1e-16) ** p["p"]


weights = np.ones_like(eff)
weights[hf <= 0.5] *= 4
weights[d < 10] *= 2
weights[a < 10] *= 2

weights[hf == 0] = 100
weights[hf == 1] = 100

fit = asb.FittedModel(
    model=model,
    x_data=hf,
    y_data=eff,
    weights=weights,
    parameter_guesses={
        "p": 2.5
    },
    residual_norm_type="L1"
)

print(fit.parameters)

fit.plot()
