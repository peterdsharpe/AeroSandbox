from aerosandbox.modeling.interpolation import InterpolatedModel
import pytest
import aerosandbox.numpy as np


def underlying_function_1D(x):
    return x ** 2 - 8 * x + 5


@pytest.fixture
def interpolated_model():
    np.random.seed(0)  # Set a seed for repeatability.

    ### Make some data
    x = np.linspace(0, 10, 11)

    return InterpolatedModel(
        x_data_coordinates=x,
        y_data_structured=underlying_function_1D(x),
    )


def test_interpolated_model_at_scalar(interpolated_model):
    assert interpolated_model(5.5) == pytest.approx(underlying_function_1D(5.5))


def test_interpolated_model_at_vector(interpolated_model):
    assert np.all(
        interpolated_model(np.array([1.5, 2.5, 3.5])) ==
        pytest.approx(underlying_function_1D(np.array([1.5, 2.5, 3.5])))
    )


def test_interpolated_model_plot(interpolated_model):
    interpolated_model.plot()


if __name__ == '__main__':
    pytest.main()
