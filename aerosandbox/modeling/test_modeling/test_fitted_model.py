from aerosandbox.modeling.fitting import fit_model
import pytest
import numpy as np

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


fitted_model = fit_model(
    model=model,
    x_data=time,
    y_data=measured_temperature,
    parameter_guesses={
        "m": 0,
        "b": 0,
    },
    residual_norm_type="LInf"
)
fitted_model.plot_fit()
