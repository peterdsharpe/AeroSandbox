from typing import Union, Dict, Any
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
                 x_data_resample: Union[int, Dict[str, Union[int, np.ndarray]]] = 10,
                 resampling_interpolator: object = interpolate.RBFInterpolator,
                 resampling_interpolator_kwargs: Dict[str, Any] = None,
                 fill_value=np.nan,  # Default behavior: return NaN for all inputs outside data range.
                 interpolated_model_kwargs: Dict[str, Any] = None,
                 ):
        """
        Creates the interpolator. Note that data must be unstructured (i.e., point cloud) for general N-dimensional
        interpolation.

        Note that if data is either 1D or structured,

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

                * If this is a dict, it must be a dict where the keys are strings matching the keys of (the
                dictionary) `x_data`. The values can either be ints or 1D np.ndarrays.

                    * If the values are ints, then that axis is linearly spaced between `min(xi)` and `max(xi)` with
                    `x_data_resample` points.

                    * If the values are 1D np.ndarrays, then those 1D np.ndarrays are used as the resampled spacing
                    for the given axis.

            resampling_interpolator: Indicates the interpolator to use in order to resample the unstructured data
            onto a structured grid. Should be analogous to scipy.interpolate.RBFInterpolator in __init__ and __call__
            syntax. See reference here:

                * https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html

            resampling_interpolator_kwargs: Indicates keyword arguments (keyword-value pairs, as a dictionary) to
            pass into the resampling interpolator.

            fill_value: Gives the value that the interpolator should return for points outside of the interpolation
            domain. The interpolation domain is defined as the hypercube bounded by the coordinates specified in
            `x_data_resample`. By default, these coordinates are the tightest axis-aligned hypercube that bounds the
            point cloud data. If fill_value is None, then the interpolator will attempt to extrapolate if the interpolation method allows.

            interpolated_model_kwargs: Indicates keyword arguments to pass into the (structured) InterpolatedModel.
            Also a dictionary. See aerosandbox.InterpolatedModel for documentation on possible inputs here.

        """
        if resampling_interpolator_kwargs is None:
            resampling_interpolator_kwargs = {}
        if interpolated_model_kwargs is None:
            interpolated_model_kwargs = {}

        try:  # Try to use the InterpolatedModel initializer. If it doesn't work, then move on.
            super().__init__(
                x_data_coordinates=x_data,
                y_data_structured=y_data,
            )
            return
        except ValueError:
            pass

        # If it didn't work, this implies that x_data is multidimensional, and hence a dict-like object. Validate this.
        try:  # Determine type of `x_data`
            x_data.keys()
            x_data.values()
            x_data.items()
        except AttributeError:
            raise TypeError("`x_data` must be a dict-like object!")

        # Make the interpolator, based on x_data and y_data.
        if resampling_interpolator == interpolate.RBFInterpolator:
            resampling_interpolator_kwargs = {
                "kernel": "thin_plate_spline",
                "degree": 1,
                **resampling_interpolator_kwargs
            }

        interpolator = resampling_interpolator(
            y=np.stack(tuple(x_data.values()), axis=1),
            d=y_data,
            **resampling_interpolator_kwargs
        )

        # If x_data_resample is an int, make it into a dict that matches x_data.
        if isinstance(x_data_resample, int):
            x_data_resample = {
                k: x_data_resample
                for k in x_data.keys()
            }

        # Now, x_data_resample should be dict-like. Validate this.
        try:
            x_data_resample.keys()
            x_data_resample.values()
            x_data_resample.items()
        except AttributeError:
            raise TypeError("`x_data_resample` must be a dict-like object!")

        # Go through x_data_resample, and replace any values that are ints with linspaced arrays.
        for k, v in x_data_resample.items():
            if isinstance(v, int):
                x_data_resample[k] = np.linspace(
                    np.min(x_data[k]),
                    np.max(x_data[k]),
                    v
                )

        x_data_coordinates: Dict = x_data_resample

        x_data_structured_values = [
            xi.flatten()
            for xi in np.meshgrid(*x_data_coordinates.values(), indexing="ij")
        ]
        x_data_structured = {
            k: xi
            for k, xi in zip(x_data.keys(), x_data_structured_values)
        }

        y_data_structured = interpolator(
            np.stack(tuple(x_data_structured_values), axis=1)
        )
        y_data_structured = y_data_structured.reshape([
            np.length(xi)
            for xi in x_data_coordinates.values()
        ])

        interpolated_model_kwargs = {
            "fill_value": fill_value,
            **interpolated_model_kwargs
        }

        super().__init__(
            x_data_coordinates=x_data_coordinates,
            y_data_structured=y_data_structured,
            **interpolated_model_kwargs,
        )

        self.x_data_raw_unstructured = x_data
        self.y_data_raw = y_data


if __name__ == '__main__':
    x = np.arange(10)
    y = x ** 3
    interp = UnstructuredInterpolatedModel(
        x_data=x,
        y_data=y
    )


    def randspace(start, stop, n=50):
        vals = (stop - start) * np.random.rand(n) + start
        vals = np.concatenate((vals[:-2], np.array([start, stop])))
        # vals = np.sort(vals)
        return vals


    np.random.seed(4)
    X = randspace(-5, 5, 200)
    Y = randspace(-5, 5, 200)
    f = np.where(X > 0, 1, 0) + np.where(Y > 0, 1, 0)
    # f = X ** 2 + Y ** 2
    interp = UnstructuredInterpolatedModel(
        x_data={
            "x": X.flatten(),
            "y": Y.flatten(),
        },
        y_data=f.flatten()
    )

    from aerosandbox.tools.pretty_plots import plt, show_plot

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # ax.plot_surface(X, Y, f, color="blue", alpha=0.2)
    ax.scatter(X.flatten(), Y.flatten(), f.flatten())
    X_plot, Y_plot = np.meshgrid(
        np.linspace(X.min(), X.max(), 500),
        np.linspace(Y.min(), Y.max(), 500),
    )
    F_plot = interp({
        "x": X_plot.flatten(),
        "y": Y_plot.flatten()
    }).reshape(X_plot.shape)
    ax.plot_surface(
        X_plot, Y_plot, F_plot,
        color="red",
        edgecolors=(1, 1, 1, 0.5),
        linewidth=0.5,
        alpha=0.2,
        rcount=40,
        ccount=40,
        shade=True,
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    import aerosandbox as asb
    import aerosandbox.numpy as np

    opti = asb.Opti()
    x = opti.variable(init_guess=0)
    y = opti.variable(init_guess=0)
    opti.minimize(interp({"x": x, "y": y}))
    sol = opti.solve()
    print(sol.value(x))
    print(sol.value(y))
