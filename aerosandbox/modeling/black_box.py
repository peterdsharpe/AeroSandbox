import inspect
from typing import Callable, Any, Union, Optional


def black_box(
        function: Callable[[Any], float],
        n_in: int = None,
        n_out: int = 1,
        fd_method: str ='forward',
        fd_step: Optional[float] = None,
        fd_step_iter: Optional[bool] = None,
) -> Callable[[Any], float]:
    """
    Wraps a function as a black box, allowing it to be used in AeroSandbox / CasADi optimization problems.

    Obtains gradients via finite differences. Assumes that the function's Jacobian is fully dense, always.

    Args:

        function:

        n_in:

        n_out:

        fd_method: One of:
            - 'forward'
            - 'backward'
            - 'central'
            - 'smoothed'

    Returns:

    """
    ### Grab the signature of the function to be wrapped - we'll need it.
    signature = inspect.signature(function)

    ### Handle default arguments.
    if n_in is None:
        n_in = len(signature.parameters)

    if n_out is None:
        n_out = 1

    ### Add limitations
    if n_out > 1:
        raise NotImplementedError("Black boxes with multiple outputs are not yet supported.")

    ### Compute finite-differencing options
    fd_options = {}

    if fd_step is not None:
        fd_options['h'] = fd_step

    if fd_step_iter is not None:
        fd_options['h_iter'] = fd_step_iter

    import casadi as cas

    class BlackBox(cas.Callback):
        def __init__(
                self,
        ):
            cas.Callback.__init__(self)
            self.construct(
                self.__class__.__name__,
                dict(
                    enable_fd=True,
                    fd_method=fd_method,
                    fd_options=fd_options,
                )
            )

        # Number of inputs and outputs
        def get_n_in(self):
            """
            Number of scalar inputs to the black-box function.
            """
            return n_in

        def get_n_out(self):
            return n_out

        # Evaluate numerically
        def eval(self, args):
            f = function(*args)
            if isinstance(f, tuple):
                return f
            else:
                return [f]

    # `wrapped_function` is a function with the same call signature as the original function, but with all arguments as positional arguments.
    wrapped_function = BlackBox()

    def wrapped_function_with_kwargs_support(*args, **kwargs):
        """
        This is a function with the same call signature as the original function, allowing both positional and keyword
        arguments. Should work identically to the original function:
            - Keyword arguments should be optional, and the default values should be the same as the original function.
            - Keyword arguments should have the option of being passed as positional arguments, provided they are in the
                correct order and all required positional arguments are passed.
            - Keyword arguments should be allowed to be passed in any order.
            - Positional arguments should be required.
            - Positional arguments should have the option of being passed as keyword arguments.
        """
        inputs = []

        # Check number of positional arguments in the signature
        n_positional_args = len(signature.parameters) - len(signature.parameters.values())
        n_args = len(signature.parameters)
        if len(args) < n_positional_args or len(args) > n_args:
            raise TypeError(
                f"Takes from {n_positional_args} to {n_args} positional arguments but {len(args)} were given"
            )

        for i, (name, parameter) in enumerate(signature.parameters.items()):

            if i < len(args):
                input = args[i]
                if name in kwargs:
                    raise TypeError(
                        f"Got multiple values for argument '{name}': {input} and {kwargs[name]}"
                    )

            elif name in kwargs:
                input = kwargs[name]

            else:
                if parameter.default is parameter.empty:
                    raise TypeError(
                        f"Missing required argument '{name}'"
                    )
                else:
                    input = parameter.default

            # print(input)

            inputs.append(input)

        return wrapped_function(*inputs)

    wrapped_function_with_kwargs_support.wrapped_function = wrapped_function
    wrapped_function_with_kwargs_support.wrapper_class = BlackBox

    return wrapped_function_with_kwargs_support


if __name__ == '__main__':

    ### Create a function that's effectively black-box (doesn't use `aerosandbox.numpy`)
    def my_func(
            a1,
            a2,
            k1=4,
            k2=5,
            k3=6,
    ):
        import math
        return (
                math.sin(a1) * math.exp(a2) * math.cos(k1 * k2) + k3
        )


    ### Now, start an optimization problem
    import aerosandbox as asb
    import aerosandbox.numpy as np

    opti = asb.Opti()

    # Wrap our function such that it can be used in an optimization problem.
    my_func_wrapped = black_box(
        function=my_func,
    )

    # Pick some variables to optimize over
    m = opti.variable(init_guess=5, lower_bound=3, upper_bound=8)
    n = opti.variable(init_guess=5, lower_bound=3, upper_bound=8)

    # Minimize the black-box function
    opti.minimize(
        my_func_wrapped(m, a2=3, k2=n)
    )

    # Solve
    sol = opti.solve()

    ### Plot the function over its inputs
    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    M, N = np.meshgrid(
        np.linspace(3, 8, 300),
        np.linspace(3, 8, 300),
    )
    fig, ax = plt.subplots()
    p.contour(
        M, N, np.vectorize(my_func)(M, a2=3, k2=N),
    )
    p.show_plot()
