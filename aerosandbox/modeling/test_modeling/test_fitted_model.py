from aerosandbox.modeling.fitting import fit_model
import pytest
import numpy as np
from dataset_temperature import time, measured_temperature

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
)
fitted_model.plot_fit()
