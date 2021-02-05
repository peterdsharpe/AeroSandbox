"""
An illustration of the various types of norms that can be used during function regression, and what they do.
"""

from aerosandbox.modeling.fitting import fit_model
import pytest
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(0)

### Create some data (some fictional system where temperature is a function of time, and we're measuring it)
n = 20

time = 100 * np.random.rand(n)

actual_temperature = 2 * time + 20  # True physics of the system

noise = 10 * np.random.randn(n)
measured_temperature = actual_temperature + noise  # Measured temperature of the system

measured_temperature[-1] -= 200  # Assume we just randomly get a bad measurement (dropout)


### Fit a model
def model(x, p):
    return p["m"] * x + p["b"]  # Linear regression


def fit_model_with_norm(residual_norm_type):
    return fit_model(
        model=model,
        x_data=time,
        y_data=measured_temperature,
        parameter_guesses={
            "m": 0,
            "b": 0,
        },
        residual_norm_type=residual_norm_type
    )


L1_model = fit_model_with_norm("L1")
L2_model = fit_model_with_norm("L2")
LInf_model = fit_model_with_norm("LInf")

### Plot fits with various different norms
x = np.linspace(0, 100)

sns.set(palette=sns.color_palette("husl", 3))
fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
plt.plot(time, measured_temperature, ".k", label="Data")
plt.plot(x, L1_model(x), label="$L_1$ Fit")
plt.plot(x, L2_model(x), label="$L_2$ Fit")
plt.plot(x, LInf_model(x), label=r"$L_\infty$ Fit")
plt.xlabel(r"Time")
plt.ylabel(r"Temperature")
plt.title(r"Illustration of Various Norm Types for Robust Regression")
plt.tight_layout()
plt.legend()
plt.show()
