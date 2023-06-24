from typing import Union, Dict
import aerosandbox.numpy as np
from aerosandbox.modeling.surrogate_model import SurrogateModel


class InterpolatedModel(SurrogateModel):
    """
    A model that is interpolated to structured (i.e., gridded) N-dimensional data. Maps from R^N -> R^1.

    You can evaluate this model at a given point by calling it just like a function, e.g.:

    >>> y = my_interpolated_model(x)

    The input to the model (`x` in the example above) is of the type:
        * in the general N-dimensional case, a dictionary where: keys are variable names and values are float/array
        * in the case of a 1-dimensional input (R^1 -> R^1), it can optionally just be a float/array.
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
                 fill_value=np.nan,  # Default behavior: return NaN for all inputs outside data range.
                 ):
        """
        Create the interpolator. Note that data must be structured (i.e., gridded on a hypercube) for general
        N-dimensional interpolation.

        Args:
            x_data_coordinates: The coordinates of each axis of the cube; essentially, the independent variable(s):

                * For the general N-dimensional case, this should be a dictionary where the keys are axis names [str]
                and the values are 1D arrays.

                * For the 1D case, you can optionally alternatively supply this as a single 1D array.

            Usage example for how you might generate this data, along with `y_data_structured`:

            >>> x1 = np.linspace(0, 5, 11)
            >>> x2 = np.linspace(0, 10, 21)
            >>> X1, X2 = np.meshgrid(x1, x2, indexing="ij")
            >>>
            >>> x_data_coordinates = {
            >>>     "x1": x1, # 1D ndarray of length 11
            >>>     "x2": x2, # 1D ndarray of length 21
            >>> }
            >>> y_data_structured = function_to_approximate(X1, X2) # 2D ndarray of shape (11, 21)

            y_data_structured: The dependent variable, expressed as a structured data "cube":

                * For the general N-dimensional case, this should be a single N-dimensional array with axis lengths
                corresponding to the inputs in `x_data_coordinates`. In the 1-dimensional case, this naturally
                reduces down to a single 1D ndarray.

                See usage example along with `x_data_coordinates` above.

            method: The method of interpolation to perform. Options:

                * "bspline" (Note: differentiable and suitable for optimization - made of piecewise-cubics. For other
                applications, other interpolators may be faster. Not monotonicity-preserving - may overshoot. Watch
                out for Runge's phenomenon; on that note, if your data is noisy, consider smoothing it first.)

                * "linear" (Note: differentiable, but not suitable for use in optimization w/o subgradient treatment due
                to C1-discontinuity)

                * "nearest" (Note: NOT differentiable, don't use in optimization. Fast.)

            fill_value: Gives the value that the interpolator should return for points outside of the interpolation
            domain. The interpolation domain is defined as the hypercube bounded by the coordinates specified in
            `x_data_coordinates`. If fill_value is None, then the interpolator will attempt to extrapolate if the interpolation method allows.

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
        if isinstance(self.x_data_coordinates, dict):
            def get_shape(value):
                if np.is_casadi_type(value, recursive=False):
                    if value.shape[1] == 1:
                        return (np.length(value),)

                try:
                    return value.shape
                except AttributeError:
                    return tuple()

            shape = np.broadcast_shapes(
                *[get_shape(v) for v in x.values()]
            )
            shape_for_reshaping = (int(np.prod(shape)),)

            def reshape(value):
                try:
                    return np.reshape(value, shape_for_reshaping)
                except ValueError:
                    if isinstance(value, int) or isinstance(value, float) or value.shape == tuple() or np.prod(
                            value.shape) == 1:
                        return value * np.ones(shape_for_reshaping)
                raise ValueError("Could not reshape value of one of the inputs!")

            x = np.stack(tuple(
                reshape(x[k])
                for k, v in self.x_data_coordinates.items()
            ), axis=1)

        output = np.interpn(
            points=self.x_data_coordinates_values,
            values=self.y_data_structured,
            xi=x,
            method=self.method,
            bounds_error=False,  # Can't be set true if general MX-type inputs are to be expected.
            fill_value=self.fill_value
        )
        try:
            return np.reshape(output, shape)
        except UnboundLocalError:
            return output
