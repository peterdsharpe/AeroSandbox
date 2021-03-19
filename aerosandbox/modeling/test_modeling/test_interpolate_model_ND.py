from aerosandbox.modeling.interpolation import InterpolatedModel
import pytest
import aerosandbox.numpy as np


def underlying_function_2D(x1, x2):
    return x1 ** 2 + (x2 + 1) ** 2


def interpolated_model():
    np.random.seed(0)  # Set a seed for repeatability.

    ### Make some data
    x1 = np.linspace(0, 10, 11)
    x2 = np.linspace(0, 10, 21)

    X1, X2 = np.meshgrid(x1, x2, indexing="ij")

    return InterpolatedModel(
        x_data_coordinates={
            "x1": x1,
            "x2": x2,
        },
        y_data_structured=underlying_function_2D(X1, X2),
    )


def test_interpolated_model_at_scalar():
    model = interpolated_model()
    x_data = {
        "x1": 5.5,
        "x2": 5.5,
    }
    assert model(x_data) == pytest.approx(underlying_function_2D(*x_data.values()))


def test_interpolated_model_at_vector():
    model = interpolated_model()
    x_data = {
        "x1": np.array([1.5, 2.5]),
        "x2": np.array([2.5, 3.5]),
    }
    assert np.all(
        model(x_data) ==
        pytest.approx(underlying_function_2D(*x_data.values()))
    )


# def test_interpolated_model_plot(interpolated_model):
#     interpolated_model.plot()


if __name__ == '__main__':
    test_interpolated_model_at_scalar()
    test_interpolated_model_at_vector()
    # pytest.main()
