from aerosandbox.modeling.interpolation import InterpolatedModel
import pytest
import aerosandbox.numpy as np


def underlying_function_1D(x):
    return x ** 2 - 8 * x + 5


def interpolated_model():
    np.random.seed(0)  # Set a seed for repeatability.

    ### Make some data
    x = np.linspace(0, 10, 11)

    return InterpolatedModel(
        x_data_coordinates=x,
        y_data_structured=underlying_function_1D(x),
    )


def test_interpolated_model_at_scalar():
    model = interpolated_model()
    assert model(5.5) == pytest.approx(underlying_function_1D(5.5))


def test_interpolated_model_at_vector():
    model = interpolated_model()
    assert np.all(
        model(np.array([1.5, 2.5, 3.5])) ==
        pytest.approx(underlying_function_1D(np.array([1.5, 2.5, 3.5])))
    )


def test_interpolated_model_plot():
    model = interpolated_model()
    model.plot()


def test_interpolated_model_zeros_patch():
    x = np.array([1, 2, 3, 4, 5])
    y = 0 * x
    f = InterpolatedModel(x, y)(3)
    assert f == pytest.approx(0)


if __name__ == '__main__':
    test_interpolated_model_zeros_patch()
    pytest.main()
