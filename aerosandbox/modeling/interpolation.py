import aerosandbox.numpy as np
from aerosandbox.modeling.surrogate_model import SurrogateModel
from typing import Union, Dict, List


class InterpolatedModel(SurrogateModel):
    """
    A model that is interpolated to structured (i.e. gridded) N-dimensional data. Maps from R^N -> R^1.

    You can evaluate this model at a given point by calling it just like a function, e.g.:

    >>> y = my_interpolated_model(x)

    The input to the model (`x` in the example above) is of the type:
        * in the general N-dimensional case, a dictionary where: keys are variable names and values are float/array
        * in the case of a 1-dimensional input (R^1 -> R^1), a float/array.
    If you're not sure what the input type of `my_interpolated_model` should be, just do:

    >>> print(my_interpolated_model) # Displays the valid input type to the model

    The output of the model (`y` in the example above) is always a float or array.

    See the docstring __init__ method of InterpolatedModel for more details of how to instantiate and use InterpolatedModel.

    One might have expected a interpolated model to be a literal Python function rather than a Python class - the
    benefit of having InterpolatedModel as a class rather than a function is that you can easily save (pickle) classes
    including data (e.g. parameters, x_data, y_data), but you can't do that with functions. And, because the
    InterpolatedModel class has a __call__ method, you can basically still just think of it like a function.

    """

    def __init__(self,
                 x_data_coordinates: Union[np.ndarray, Dict[str, np.ndarray]],
                 y_data_structured: np.ndarray,
                 method: str = "bspline",
                 fill_value=np.NaN,  # Default behavior NaNs outside range
                 ):
        """

        Args:
            x_data_coordinates: # TODO

            y_data_structured: # TODO

            method: The method of interpolation to perform. If you want it to be differentiable, your options are:

                * "linear"

                * "bspline"

            bounds_error: If True, when interpolated values are requested outside of the domain of the input data,
            a ValueError is raised. If False, then fill_value is used.

            fill_value: If provided, the value to use for points outside of the interpolation domain. If None,
            values outside the domain are extrapolated.

        """
        try:
            x_data_coordinates_values = x_data_coordinates.values()
        except AttributeError:  # If x_data_coordinates is not a dict
            x_data_coordinates_values = tuple([x_data_coordinates])

        ### Validate inputs
        for coordinates in x_data_coordinates_values:
            if len(coordinates.shape) != 1:
                raise ValueError("""
                    `x_data_coordinates` must be either: 
                        * In the general N-dimensional case, a dict where values are 1D ndarrays defining the coordinates of each axis.
                        * In the 1D case, can also be a 1D ndarray.
                    """)
        implied_y_data_shape = tuple(len(coordinates) for coordinates in x_data_coordinates_values)
        if not y_data_structured.shape == implied_y_data_shape:
            raise ValueError(f"""
            The shape of `y_data_structured` should be {implied_y_data_shape}
            """)

        ### Store data
        self.x_data_coordinates = x_data_coordinates
        self.x_data_coordinates_values = x_data_coordinates_values
        self.y_data_structured = y_data_structured
        self.method = method
        self.fill_value = fill_value

        ### Create unstructured versions of the data for plotting, etc.
        x_data = x_data_coordinates
        if isinstance(x_data, dict):
            x_data_values = np.meshgrid(*x_data_coordinates_values, indexing="ij")
            x_data = {
                k: v.reshape(-1)
                for k, v in zip(x_data_coordinates.keys(), x_data_values)
            }
        self.x_data = x_data
        self.y_data = np.ravel(y_data_structured, order="F")

    def __call__(self, x):
        return np.interpn(
            points=self.x_data_coordinates_values,
            values=self.y_data_structured,
            xi=x,
            method=self.method,
            bounds_error=False,  # Can't be set true if general MX-type inputs are to be expected.
            fill_value=self.fill_value
        )

        # TODO finish
