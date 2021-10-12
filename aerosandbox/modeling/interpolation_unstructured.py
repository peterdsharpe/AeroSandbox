from typing import Union, Dict
import aerosandbox.numpy as np
from aerosandbox.modeling.interpolation import InterpolatedModel
from scipy import interpolate

class UnstructuredInterpolatedModel(InterpolatedModel):
    """
    A model that is interpolated to unstructured (i.e., point cloud) N-dimensional data. Maps from R^N -> R^1.

    You can evaluate this model at a given point by calling it just like a function, e.g.:

    >>> y = my_interpolated_model(x)

    The input to the model (`x` in the example above) is of the type:
        * in the general N-dimensional case, a dictionary where: keys are variable names and values are float/array
        * in the case of a 1-dimensional input (R^1 -> R^1), it can optionally just be a float/array.
    If you're not sure what the input type of `my_interpolated_model` should be, just do:

    >>> print(my_interpolated_model) # Displays the valid input type to the model

    The output of the model (`y` in the example above) is always a float or array.

    See the docstring __init__ method of InterpolatedModel for more details of how to instantiate and use UnstructuredInterpolatedModel.

    """

    def __init__(self,
                 x_data: Union[np.ndarray, Dict[str, np.ndarray]],
                 y_data: np.ndarray,
                 x_data_resample: Union[int, np.ndarray, Dict[str, Union[int, np.ndarray]]] = 20,
                 data_resample_method: str = "rbf",
                 method: str = "bspline-resample",
                 ):
        """
        Creates the interpolator. Note that data must be unstructured (i.e., point cloud) for general N-dimensional
        interpolation.

        Args:

            x_data: Values of the dependent variable(s) in the dataset to be fitted. This is a dictionary; syntax is {
            var_name:var_data}.

                * If the model is one-dimensional (e.g. f(x1) instead of f(x1, x2, x3...)), you can instead supply x_data
                as a 1D ndarray. (If you do this, just treat `x` as an array in your model, not a dict.)

            y_data: Values of the independent variable in the dataset to be fitted. [1D ndarray of length n]

            x_data_resample: A parameter that guides how the x_data should be resampled onto a structured grid.

                * If this is an int, we look at each axis of the `x_data` (here, we'll call this `xi`),
                and we resample onto a linearly-spaced grid between `min(xi)` and `max(xi)` with `x_data_resample`
                points.

                * If `x_data` is one-dimensional and provided as a np.ndarray, then x_data_resample can also be a
                np.ndarray. However this use case doesn't really make much sense - if you have 1D data, then there is
                no difference between structured and unstructured data, and you should probably just use
                InterpolatedModel instead. If np.ndarray

                * If this is a dict, it must be a dict where the keys are strings matching the keys of (the
                dictionary) `x_data`. The values can either be ints or np.ndarrays.

                    * If the values are ints, then that axis is linearly spaced between `min(xi)` and `max(xi)` with
                    `x_data_resample` points.

                    * If the values are np.ndarrays, then those np.ndarrays are used as the resampled spacing.

            method:

        """
        pass