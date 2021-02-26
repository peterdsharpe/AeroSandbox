"""
An illustration of the various types of norms that can be used during function regression, and what they look like.
"""

from aerosandbox.modeling.fitting import FittedModel
import pytest
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataset_temperature import time, measured_temperature


def test_fit_model_norm_type(plot=False):
    ### Fit a model
    def model(x, p):
        return p["1"] * x + p["0"]  # Linear regression

    def fit_model_with_norm(residual_norm_type):
        return FittedModel(
            model=model,
            x_data=time,
            y_data=measured_temperature,
            parameter_guesses={
                "1": 0,
                "0": 0,
            },
            residual_norm_type=residual_norm_type
        )

    L1_model = fit_model_with_norm("L1")
    L2_model = fit_model_with_norm("L2")
    LInf_model = fit_model_with_norm("LInf")

    if plot:
        ### Plot fits with various different norms
        x = np.linspace(0, 100)

        sns.set(palette=sns.color_palette("husl", 3))
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
        plt.plot(time, measured_temperature, ".k", label="Data")
        plt.plot(x, L1_model(x), label=r"$L_1$ Fit: $\min (\sum |e|)$")
        plt.plot(x, L2_model(x), label=r"$L_2$ Fit: $\min (\sum e^2)$")
        plt.plot(x, LInf_model(x), label=r"$L_\infty$ Fit: $\min (\max |e|)$")
        plt.xlabel(r"Time")
        plt.ylabel(r"Temperature")
        plt.title(r"Illustration of Various Norm Types for Robust Regression (Quadratic Model)")
        plt.tight_layout()
        plt.legend()
        plt.show()


if __name__ == '__main__':
    test_fit_model_norm_type(plot=True)
