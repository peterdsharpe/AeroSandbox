from aerosandbox.modeling.fitting import FittedModel
import pytest
import aerosandbox.numpy as np
from dataset_temperature import time, measured_temperature


@pytest.fixture
def get_fitted_model():
    ### Fit a model
    def model(x, p):
        return p["m"] * x + p["b"]  # Linear regression

    fitted_model = FittedModel(
        model=model,
        x_data=time,
        y_data=measured_temperature,
        parameter_guesses={
            "m": 0,
            "b": 0,
        },
    )
    return fitted_model


def test_plot(get_fitted_model):
    get_fitted_model.plot()


if __name__ == '__main__':
    pytest.main()
