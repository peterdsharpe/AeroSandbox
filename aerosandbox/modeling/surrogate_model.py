from aerosandbox.common import AeroSandboxObject
from abc import abstractmethod
from typing import Union, Dict, List, Tuple
import aerosandbox.numpy as np


class SurrogateModel(AeroSandboxObject):
    """
    A SurrogateModel is effectively a callable; it only has the __call__ method, and all subclasses must explicitly
    overwrite this. The only reason it is not a callable is that you want to be able to save it to disk (via
    pickling) while also having the capability to save associated data (for example, constants associated with a
    particular model, or underlying data).

    If data is used to generate the SurrogateModel, it should be stored as follows:

        * The independent variable(s) should be stored as SurrogateModel.x_data

            * in the general N-dimensional case, x_data should be a dictionary where: keys are variable names and
            values are float/array

            * in the case of a 1-dimensional input (R^1 -> R^1), x-data should be a float/array.

        * The dependent variable should be stored as SurrogateModel.y_data

            * The type of this variable should be a float or an np.ndarray.

    Even if you don't have any real x_data or y_data to add as SurrogateModel.x_data or SurrogateModel.y_data,
        it's recommended (but not required) that you add values here as examples that users can inspect in order to see
        the data types and array shapes required.
    """

    @abstractmethod
    def __init__(self):
        """
        SurrogateModel is an abstract class; you should not instantiate it directly.
        """
        pass

    @abstractmethod  # If you subclass SurrogateModel, you must overwrite __call__ so that it's a callable.
    def __call__(self,
                 x: Union[int, float, np.ndarray, Dict[str, np.ndarray]]
                 ) -> Union[float, np.ndarray]:
        """
        Evaluates the surrogate model at some given input x.

        The input `x` is of the type:
            * in the general N-dimensional case, a dictionary where keys are variable names and values are float/array.
            * in the case of a 1-dimensional input (R^1 -> R^2), a float/array.

        """
        ### Perform basic type checking on x, if x_data exists as a reference.
        try:
            x_data_is_dict = isinstance(self.x_data, dict)
            input_is_dict = isinstance(x, dict)

            if x_data_is_dict and not input_is_dict:
                raise TypeError(
                    f"The input to this model should be a dict with: keys {self.input_names()}, values as float or array.")

            if input_is_dict and not x_data_is_dict:
                raise TypeError("The input to this model should be a float or array.")
        except NameError:  # If x_data does not exist
            pass

    def __repr__(self) -> str:
        input_names = self.input_names()
        if input_names is not None:
            input_description = f"a dict with: keys {input_names}, values as float or array"
        else:
            input_description = f"a float or array"
        return "\n".join([
            f"SurrogateModel(x) [R^{self.input_dimensionality()} -> R^1]",
            f"\tInput: {input_description}",
            f"\tOutput: float or array",
        ])

    def input_dimensionality(self) -> int:
        """
        Returns the number of inputs that should be supplied in x, where x is the input to the SurrogateModel.
        """
        input_names = self.input_names()
        if input_names is not None:
            return len(input_names)
        else:
            return 1

    def input_names(self) -> Union[List, None]:
        """
        If x (the input to this model) is supposed to be a dict, this method returns the keys that should be part of x.

        If x is 1D and simply takes in floats or arrays, or if no x_data exists, returns None.
        """
        try:
            return list(self.x_data.keys())
        except AttributeError:
            return None

    def plot(self, resolution=250):
        import matplotlib.pyplot as plt

        def axis_range(x_data_axis: np.ndarray) -> Tuple[float, float]:
            """
            Given the entries of one axis of the dependent variable, determine a min/max range over which to plot the fit.
            Args:
                x_data_axis: The entries of one axis of the dependent variable, i.e. x_data["x1"].

            Returns: A tuple representing the (min, max) value over which to plot that axis.
            """
            minval = np.min(x_data_axis)
            maxval = np.max(x_data_axis)

            return (minval, maxval)

        if self.input_dimensionality() == 1:

            ### Parse the x_data
            if self.input_names() is not None:
                x_name = self.x_data.keys()[0]
                x_data = self.x_data.values()[0]

                minval, maxval = axis_range(x_data)

                x_fit = {x_name: np.linspace(minval, maxval, resolution)}
                y_fit = self(x_fit)
            else:
                x_name = "x"
                x_data = self.x_data

                minval, maxval = axis_range(x_data)

                x_fit = np.linspace(minval, maxval, resolution)
                y_fit = self(x_fit)

            ### Plot the 2D figure
            fig = plt.figure(dpi=200)
            plt.plot(
                x_data,
                self.y_data,
                ".k",
                label="Data",
            )
            plt.plot(
                x_fit,
                y_fit,
                "-",
                color="#cb3bff",
                label="Fit",
                zorder=4,
            )
            plt.xlabel(x_name)
            plt.ylabel(rf"$f({x_name})$")
            plt.title(r"Fit of FittedModel")
            plt.tight_layout()
            plt.legend()
            plt.show()

        else:
            raise NotImplementedError()
