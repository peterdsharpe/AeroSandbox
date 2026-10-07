import aerosandbox.numpy as np
from aerosandbox.modeling.surrogate_model import SurrogateModel
from aerosandbox.modeling.interpolation import InterpolatedModel
import pytest

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend, so that plot tests don't block.


def test_call_works_without_x_data():
    """
    Regression test: SurrogateModel.__call__'s type-check guard used to catch NameError,
    but a missing `x_data` attribute raises AttributeError - so subclasses without x_data
    (explicitly allowed by the class docstring) crashed on every call.
    """

    class NoDataModel(SurrogateModel):
        def __init__(self):
            pass

        def __call__(self, x):
            super().__call__(x)
            return x * 2

    model = NoDataModel()
    assert model(3.0) == pytest.approx(6.0)


def test_call_type_checking_still_works():
    """The type-check guard should still fire when x_data does exist."""

    class DictDataModel(SurrogateModel):
        def __init__(self):
            self.x_data = {"x": np.linspace(0, 10, 11)}

        def __call__(self, x):
            super().__call__(x)
            return x["x"] * 2

    model = DictDataModel()
    assert model({"x": 3.0}) == pytest.approx(6.0)

    with pytest.raises(TypeError):
        model(5.0)  # Should be a dict, since x_data is a dict

    class ArrayDataModel(SurrogateModel):
        def __init__(self):
            self.x_data = np.linspace(0, 10, 11)

        def __call__(self, x):
            super().__call__(x)
            return x * 2

    model_1D = ArrayDataModel()
    assert model_1D(3.0) == pytest.approx(6.0)

    with pytest.raises(TypeError):
        model_1D({"x": 5.0})  # Should be a float/array, since x_data is an array


def test_plot_1D_dict_x_data(monkeypatch):
    """
    Regression test: plot() used to crash with "'dict_keys' object is not subscriptable"
    for any 1D model with dict x_data.
    """
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    x = np.linspace(0, 10, 11)
    model = InterpolatedModel(
        x_data_coordinates={"x": x},
        y_data_structured=x**2,
    )

    model.plot()
    plt.close("all")


def test_plot_1D_array_x_data(monkeypatch):
    """plot() with plain-array x_data should also work."""
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    x = np.linspace(0, 10, 11)
    model = InterpolatedModel(
        x_data_coordinates=x,
        y_data_structured=x**2,
    )

    model.plot()
    plt.close("all")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
