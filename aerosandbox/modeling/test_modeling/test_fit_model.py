from aerosandbox.modeling.fitting import FittedModel
import pytest
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(palette=sns.color_palette("husl"))

plot = False  # Should we plot during testing?


def test_single_dimensional_polynomial_fitting():
    np.random.seed(0)  # Set a seed for repeatability.

    ### Make some data
    x = np.linspace(0, 10, 50)
    noise = 5 * np.random.randn(len(x))
    y = x ** 2 - 8 * x + 5 + noise

    ### Fit data
    def model(x, p):
        return (
                p["a"] * x["x1"] ** 2 +
                p["b"] * x["x1"] +
                p["c"]
        )

    x_data = {
        "x1": x
    }

    fitted_model = FittedModel(
        model=model,
        x_data=x_data,
        y_data=y,
        parameter_guesses={
            "a": 1,
            "b": 1,
            "c": 1,
        },
        parameter_bounds={
            "a": (None, None),
            "b": (None, None),
            "c": (None, None),
        },
    )

    ### Plot data and fit
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
        plt.plot(x, y, ".", label="Data")
        plt.plot(x, model(x_data, fitted_parameters), "-", label="Fit")
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.title(r"Automatic-Differentiable Fitting")
        plt.tight_layout()
        plt.legend()
        # plt.savefig("C:/Users/User/Downloads/temp.svg")
        plt.show()

    ### Check that the fit is right
    assert fitted_model.parameters["a"] == pytest.approx(1.046091, abs=1e-3)
    assert fitted_model.parameters["b"] == pytest.approx(-9.166716, abs=1e-3)
    assert fitted_model.parameters["c"] == pytest.approx(9.984351, abs=1e-3)


def test_multidimensional_power_law_fitting():
    np.random.seed(0)  # Set a seed for repeatability.

    ### Make some data z(x,y)
    x = np.logspace(0, 3)
    y = np.logspace(0, 3)
    X, Y = np.meshgrid(x, y, indexing="ij")
    noise = np.random.lognormal(mean=0, sigma=0.05)
    Z = 0.5 * X ** 0.75 * Y ** 1.25 * noise

    ### Fit data
    def model(x, p):
        return (
                p["multiplier"] *
                x["X"] ** p["X_power"] *
                x["Y"] ** p["Y_power"]
        )

    x_data = {
        "X": X.flatten(),
        "Y": Y.flatten(),
    }

    fitted_model = FittedModel(
        model=model,
        x_data=x_data,
        y_data=Z.flatten(),
        parameter_guesses={
            "multiplier": 1,
            "X_power"   : 1,
            "Y_power"   : 1,
        },
        parameter_bounds={
            "multiplier": (None, None),
            "X_power"   : (None, None),
            "Y_power"   : (None, None),
        },
        put_residuals_in_logspace=True
        # Putting residuals in logspace minimizes the norm of log-error instead of absolute error
    )

    ### Check that the fit is right
    assert fitted_model.parameters["multiplier"] == pytest.approx(0.546105, abs=1e-3)
    assert fitted_model.parameters["X_power"] == pytest.approx(0.750000, abs=1e-3)
    assert fitted_model.parameters["Y_power"] == pytest.approx(1.250000, abs=1e-3)


def test_Linf_single_dimensional_fit():
    np.random.seed(0)  # Set a seed for repeatability

    ### Making data
    hour = np.linspace(1, 10, 100)
    noise = 0.1 * np.random.randn(len(hour))
    temperature_c = np.log(hour) + noise

    ### Fit
    def model(x, p):
        return p["m"] * x["hour"] + p["b"]

    x_data = {
        "hour": hour,
    }
    y_data = temperature_c
    fitted_model = FittedModel(
        model=model,
        x_data=x_data,
        y_data=y_data,
        parameter_guesses={
            "m": 0,
            "b": 0,
        },
        residual_norm_type="Linf",
    )

    # Check that the fit is right
    assert fitted_model.parameters["m"] == pytest.approx(0.247116, abs=1e-5)
    assert fitted_model.parameters["b"] == pytest.approx(0.227797, abs=1e-5)


def test_Linf_without_x_in_dict():
    np.random.seed(0)  # Set a seed for repeatability

    ### Making data
    hour = np.linspace(1, 10, 100)
    noise = 0.1 * np.random.randn(len(hour))
    temperature_c = np.log(hour) + noise

    ### Fit
    def model(x, p):
        return p["m"] * x + p["b"]

    x_data = hour
    y_data = temperature_c

    fitted_model = FittedModel(
        model=model,
        x_data=x_data,
        y_data=y_data,
        parameter_guesses={
            "m": 0,
            "b": 0,
        },
        residual_norm_type="Linf",
    )

    # Check that the fit is right
    assert fitted_model.parameters["m"] == pytest.approx(0.247116, abs=1e-5)
    assert fitted_model.parameters["b"] == pytest.approx(0.227797, abs=1e-5)

def test_type_errors():
    np.random.seed(0)  # Set a seed for repeatability

    ### Making data
    hour = np.linspace(1, 10, 100)
    noise = 0.1 * np.random.randn(len(hour))
    temperature_c = np.log(hour) + noise

    ### Fit
    def model(x, p):
        return p["m"] * x + p["b"]

    x_data = hour
    y_data = temperature_c

    fitted_model = FittedModel(
        model=model,
        x_data=x_data,
        y_data=y_data,
        parameter_guesses={
            "m": 0,
            "b": 0,
        },
        residual_norm_type="Linf",
    )

    fitted_model(5)

    with pytest.raises(TypeError):
        fitted_model({
            "temperature": 5
        })

    def model(x, p):
        return p["m"] * x["hour"] + p["b"]

    fitted_model = FittedModel(
        model=model,
        x_data={
            "hour": hour
        },
        y_data=y_data,
        parameter_guesses={
            "m": 0,
            "b": 0,
        },
        residual_norm_type="Linf",
    )

    fitted_model({
        "hour": 5
    })

    with pytest.raises(TypeError):
        fitted_model(5)

if __name__ == '__main__':
    pytest.main()
