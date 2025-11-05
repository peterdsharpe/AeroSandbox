import aerosandbox.numpy as np
from aerosandbox.modeling.interpolation import InterpolatedModel
import pytest


def test_interpolation_1d_linear():
    """Test 1D linear interpolation."""
    x_data = np.array([0, 1, 2, 3, 4])
    y_data = 2 * x_data + 1  ### y = 2x + 1

    model = InterpolatedModel(
        x_data_coordinates={"x": x_data},
        y_data_structured=y_data,
    )

    ### Test at data points (must use dict when x_data_coordinates is a dict)
    for x_val, y_expected in zip(x_data, y_data):
        y_pred = model({"x": x_val})
        assert np.abs(y_pred - y_expected) < 0.1

    ### Test interpolation
    y_pred = model({"x": 1.5})
    expected = 2 * 1.5 + 1
    assert np.abs(y_pred - expected) < 0.1


def test_interpolation_1d_quadratic():
    """Test 1D interpolation with quadratic function."""
    x_data = np.linspace(0, 10, 50)
    y_data = x_data**2 + 3 * x_data + 2

    model = InterpolatedModel(
        x_data_coordinates={"x": x_data},
        y_data_structured=y_data,
    )

    ### Test at intermediate point
    x_test = 5.5
    y_pred = model({"x": x_test})
    y_expected = x_test**2 + 3 * x_test + 2

    ### Linear interpolation won't be exact for quadratic, but should be close
    assert np.abs(y_pred - y_expected) < 1.0


def test_interpolation_2d():
    """Test 2D interpolation."""
    x_data = np.linspace(0, 5, 10)
    y_data = np.linspace(0, 5, 10)

    X, Y = np.meshgrid(x_data, y_data, indexing="ij")
    Z_data = X**2 + Y**2  ### z = x^2 + y^2

    model = InterpolatedModel(
        x_data_coordinates={"x": x_data, "y": y_data},
        y_data_structured=Z_data,
    )

    ### Test at a grid point
    z_pred = model({"x": 2.0, "y": 2.0})
    expected = 2.0**2 + 2.0**2
    assert np.abs(z_pred - expected) < 0.5


def test_interpolation_extrapolation():
    """Test that extrapolation returns NaN by default."""
    x_data = np.array([1, 2, 3, 4])
    y_data = np.array([2, 4, 6, 8])

    model = InterpolatedModel(
        x_data_coordinates={"x": x_data},
        y_data_structured=y_data,
    )

    ### Extrapolate beyond data range (default fill_value=np.nan)
    y_pred = model({"x": 5.0})

    ### Should return NaN for out-of-bounds by default
    assert np.isnan(y_pred)


def test_interpolation_single_point():
    """Test that interpolation with single data point raises error (not supported)."""
    x_data = np.array([1.0])
    y_data = np.array([5.0])

    ### CasADi requires at least 2 grid points - this should raise an error
    with pytest.raises((RuntimeError, ValueError)):
        model = InterpolatedModel(
            x_data_coordinates={"x": x_data},
            y_data_structured=y_data,
        )
        model({"x": 1.0})


def test_interpolation_monotonic_data():
    """Test interpolation preserves monotonicity where expected."""
    x_data = np.linspace(0, 10, 20)
    y_data = np.exp(x_data)  ### Monotonically increasing

    model = InterpolatedModel(
        x_data_coordinates={"x": x_data},
        y_data_structured=y_data,
    )

    ### Test at several points
    x_test = np.linspace(1, 9, 10)
    y_pred = [model({"x": x}) for x in x_test]

    ### Predictions should be monotonically increasing
    assert all(y_pred[i] < y_pred[i + 1] for i in range(len(y_pred) - 1))


def test_interpolation_constant_function():
    """Test interpolation of constant function."""
    x_data = np.linspace(0, 10, 20)
    y_data = np.ones_like(x_data) * 5.0

    model = InterpolatedModel(
        x_data_coordinates={"x": x_data},
        y_data_structured=y_data,
    )

    ### Test at various points
    for x in [0, 2.5, 5.0, 7.5, 10.0]:
        y_pred = model({"x": x})
        assert np.isclose(y_pred, 5.0, atol=1e-6)


def test_interpolation_negative_values():
    """Test interpolation with negative values."""
    x_data = np.linspace(-5, 5, 20)
    y_data = x_data**3  ### Includes negative values

    model = InterpolatedModel(
        x_data_coordinates={"x": x_data},
        y_data_structured=y_data,
    )

    y_pred = model({"x": -2.5})
    expected = (-2.5) ** 3

    ### Should handle negative values
    assert np.abs(y_pred - expected) < 10


def test_interpolation_oscillatory_function():
    """Test interpolation of oscillatory function."""
    x_data = np.linspace(0, 2 * np.pi, 50)
    y_data = np.sin(x_data)

    model = InterpolatedModel(
        x_data_coordinates={"x": x_data},
        y_data_structured=y_data,
    )

    ### Test at pi/2 (should be close to 1)
    y_pred = model({"x": np.pi / 2})
    assert np.abs(y_pred - 1.0) < 0.1


def test_interpolation_steep_gradient():
    """Test interpolation with steep gradients."""
    x_data = np.linspace(0, 5, 50)
    y_data = np.exp(2 * x_data)  ### Steep exponential

    model = InterpolatedModel(
        x_data_coordinates={"x": x_data},
        y_data_structured=y_data,
    )

    y_pred = model({"x": 2.5})
    expected = np.exp(2 * 2.5)

    ### May not be super accurate due to steepness, but should be in ballpark
    assert np.abs(y_pred - expected) / expected < 0.5


def test_interpolation_2d_separable():
    """Test 2D interpolation with separable function."""
    x_data = np.linspace(0, 5, 10)
    y_data = np.linspace(0, 3, 10)

    X, Y = np.meshgrid(x_data, y_data, indexing="ij")
    Z_data = X * Y  ### Separable: f(x,y) = x * y

    model = InterpolatedModel(
        x_data_coordinates={"x": x_data, "y": y_data},
        y_data_structured=Z_data,
    )

    z_pred = model({"x": 2.5, "y": 1.5})
    expected = 2.5 * 1.5

    assert np.abs(z_pred - expected) < 0.5


def test_interpolation_boundary_points():
    """Test interpolation at boundary points."""
    x_data = np.linspace(0, 10, 20)
    y_data = x_data**2

    model = InterpolatedModel(
        x_data_coordinates={"x": x_data},
        y_data_structured=y_data,
    )

    ### Test at boundaries
    y_pred_min = model({"x": x_data[0]})
    y_pred_max = model({"x": x_data[-1]})

    assert np.isclose(y_pred_min, y_data[0], atol=1e-6)
    assert np.isclose(y_pred_max, y_data[-1], atol=1e-6)


def test_interpolation_irregular_spacing():
    """Test interpolation with irregularly spaced data."""
    x_data = np.array([0, 0.5, 1.5, 3.0, 5.0, 8.0, 10.0])
    y_data = x_data**2

    model = InterpolatedModel(
        x_data_coordinates={"x": x_data},
        y_data_structured=y_data,
    )

    ### Test interpolation between irregular points
    y_pred = model({"x": 2.0})

    ### Should be somewhere reasonable
    assert 1.5**2 < y_pred < 3.0**2


def test_interpolation_vector_output():
    """Test that vector output raises an error (not supported by InterpolatedModel)."""
    x_data = np.linspace(0, 5, 20)
    y_data = np.stack([x_data, x_data**2, x_data**3], axis=-1)  ### Shape: (20, 3)

    ### InterpolatedModel only supports scalar outputs
    with pytest.raises(ValueError):
        model = InterpolatedModel(
            x_data_coordinates={"x": x_data},
            y_data_structured=y_data,
        )


def test_interpolation_3d():
    """Test 3D interpolation."""
    x_data = np.linspace(0, 2, 5)
    y_data = np.linspace(0, 2, 5)
    z_data = np.linspace(0, 2, 5)

    X, Y, Z = np.meshgrid(x_data, y_data, z_data, indexing="ij")
    W_data = X + Y + Z  ### Simple sum

    model = InterpolatedModel(
        x_data_coordinates={"x": x_data, "y": y_data, "z": z_data},
        y_data_structured=W_data,
    )

    w_pred = model({"x": 1.0, "y": 1.0, "z": 1.0})
    expected = 3.0

    assert np.abs(w_pred - expected) < 0.5


def test_interpolation_zero_values():
    """Test interpolation when data contains zeros."""
    x_data = np.linspace(-5, 5, 20)
    y_data = x_data  ### Crosses zero

    model = InterpolatedModel(
        x_data_coordinates={"x": x_data},
        y_data_structured=y_data,
    )

    y_pred = model({"x": 0})
    assert np.abs(y_pred) < 0.5


def test_interpolation_large_dataset():
    """Test interpolation with large dataset."""
    x_data = np.linspace(0, 100, 1000)
    y_data = np.sin(x_data / 10)

    model = InterpolatedModel(
        x_data_coordinates={"x": x_data},
        y_data_structured=y_data,
    )

    y_pred = model({"x": 50})
    expected = np.sin(50 / 10)

    assert np.abs(y_pred - expected) < 0.1


def test_interpolation_small_range():
    """Test interpolation over small range."""
    x_data = np.linspace(0.001, 0.002, 10)
    y_data = x_data * 1000

    model = InterpolatedModel(
        x_data_coordinates={"x": x_data},
        y_data_structured=y_data,
    )

    y_pred = model({"x": 0.0015})
    expected = 1.5

    assert np.abs(y_pred - expected) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
