from aerosandbox.modeling.interpolation_unstructured import (
    UnstructuredInterpolatedModel,
)
import aerosandbox.numpy as np
import pytest


def underlying_function_1D(x):
    return x**2


def test_unsorted_1D_array_input():
    """
    Regression test: unsorted 1D point-cloud data used to construct successfully, then crash
    at call time with a CasADi "Gridpoints must be strictly increasing" error.
    """
    x = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
    y = underlying_function_1D(x)

    model = UnstructuredInterpolatedModel(x_data=x, y_data=y)

    assert model(2.5) == pytest.approx(underlying_function_1D(2.5), abs=0.5)


def test_unsorted_1D_dict_input():
    """Same as test_unsorted_1D_array_input, but with x_data supplied as a single-key dict."""
    x = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
    y = underlying_function_1D(x)

    model = UnstructuredInterpolatedModel(x_data={"x": x}, y_data=y)

    assert model({"x": 2.5}) == pytest.approx(underlying_function_1D(2.5), abs=0.5)


def test_duplicate_1D_x_data_raises_clear_error():
    """Duplicate x-coordinates in a 1D dataset should raise a clear ValueError at construction."""
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        UnstructuredInterpolatedModel(
            x_data=np.array([1.0, 2.0, 2.0, 3.0]),
            y_data=np.array([1.0, 4.0, 4.0, 9.0]),
        )


def test_1D_shortcut_path_respects_fill_value_and_kwargs():
    """
    Regression test: the 1D/structured shortcut path used to silently discard `fill_value`
    and `interpolated_model_kwargs`, and never set the raw-data attributes.
    """
    x = np.linspace(0, 9, 10)
    y = underlying_function_1D(x)

    ### Default behavior: fill_value=np.nan, so out-of-domain queries return NaN.
    model_default = UnstructuredInterpolatedModel(x_data=x, y_data=y)
    assert np.isnan(model_default(15.0))

    ### fill_value=None should allow extrapolation (i.e., not return NaN).
    model_extrapolate = UnstructuredInterpolatedModel(
        x_data=x, y_data=y, fill_value=None
    )
    assert not np.isnan(model_extrapolate(15.0))

    ### interpolated_model_kwargs should be passed through.
    model_linear = UnstructuredInterpolatedModel(
        x_data=x, y_data=y, interpolated_model_kwargs={"method": "linear"}
    )
    assert model_linear.method == "linear"

    ### Raw-data attributes should be set on the shortcut path, and be point-paired.
    assert np.allclose(
        model_default.y_data_raw,
        underlying_function_1D(model_default.x_data_raw_unstructured),
    )


def test_x_data_resample_dict_not_mutated():
    """
    Regression test: the constructor used to replace int entries of the user-supplied
    x_data_resample dict with linspaced arrays, mutating the caller's dict in-place.
    """
    np.random.seed(0)
    X = np.random.rand(50) * 10
    Y = np.random.rand(50) * 10
    F = X + Y

    x_data_resample = {"x": 5, "y": 7}
    UnstructuredInterpolatedModel(
        x_data={"x": X, "y": Y},
        y_data=F,
        x_data_resample=x_data_resample,
    )

    assert x_data_resample == {"x": 5, "y": 7}


def test_ND_point_cloud_still_works():
    """Sanity check that the N-dimensional (resampling) code path is unaffected."""
    np.random.seed(0)
    X = np.random.rand(50) * 10
    Y = np.random.rand(50) * 10
    F = X + Y

    model = UnstructuredInterpolatedModel(
        x_data={"x": X, "y": Y},
        y_data=F,
    )

    assert model({"x": 5.0, "y": 5.0}) == pytest.approx(10, abs=0.5)

    ### fill_value / interpolated_model_kwargs should still be respected on this path.
    model_linear = UnstructuredInterpolatedModel(
        x_data={"x": X, "y": Y},
        y_data=F,
        fill_value=None,
        interpolated_model_kwargs={"method": "linear"},
    )
    assert model_linear.method == "linear"
    assert model_linear.fill_value is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
