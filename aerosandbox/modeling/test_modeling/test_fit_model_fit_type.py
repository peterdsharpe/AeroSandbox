"""
An illustration of the various fit types that can be used during function regression, and what they look like.
"""

from aerosandbox.modeling.fitting import fit_model
import pytest
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataset_temperature import time, measured_temperature


def test_fit_model_fit_type(plot=False):
    ### Fit a model
    def model(x, p):
        return p["a"] * x ** 2 + p["b"] * x + p["c"]  # Quadratic regression

    def fit_model_with_fit_type(fit_type):
        return fit_model(
            model=model,
            x_data=time,
            y_data=measured_temperature,
            parameter_guesses={
                "a": 0,
                "b": 0,
                "c": 0,
            },
            fit_type=fit_type,
            residual_norm_type="L1"
        )

    best_fit_model = fit_model_with_fit_type("best")
    upper_bound_model = fit_model_with_fit_type("upper bound")
    lower_bound_model = fit_model_with_fit_type("lower bound")

    if plot:
        ### Plot fits with various different norms
        x = np.linspace(0, 100)

        sns.set(palette=sns.color_palette("husl", 3))
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
        plt.plot(time, measured_temperature, ".k", label="Data")
        plt.plot(x, best_fit_model(x), label=r"Best Fit")
        plt.plot(x, upper_bound_model(x), label=r"Upper-Bound Fit")
        plt.plot(x, lower_bound_model(x), label=r"Lower-Bound Fit")
        plt.xlabel(r"Time")
        plt.ylabel(r"Temperature")
        plt.title(r"Illustration of Fit Types for Robust Surrogate Modeling")
        plt.tight_layout()
        plt.legend()
        plt.show()


if __name__ == '__main__':
    test_fit_model_fit_type(True)
