from aerosandbox.tools import inspect_tools
from typing import Union, Tuple, List
import aerosandbox.numpy as np
import warnings


def qp(
        *args: Tuple[Union[np.ndarray, List]],
        backend="plotly",
        show=True,
        plotly_renderer: Union[str, None] = "browser",
        orthographic=True,
        stacklevel=1
) -> None:
    """
    Quickly plots ("QP") a 1D, 2D, or 3D dataset as a line plot with markers. Useful for exploratory data analysis.

    Example:

        >>> import aerosandbox.numpy as np
        >>>
        >>> x = np.linspace(0, 10)
        >>> y = x ** 2
        >>> z = np.sin(x)
        >>> qp(x, y, z)

    Args:
        *args: The arguments that you want to plot. You can provide 1, 2, or 3 arrays, all of which should be 1D and of the same length.

        backend: The backend to use. Current options:
            * "plotly"

        show: A boolean of whether or not to show the plot.

        plotly_renderer: A string of what to use as the Plotly renderer. If you don't want to overwrite a default that you've already set, set this variable to None.

        orthographic: A boolean of whether or not to use an orthographic (rather than persepctive) projection when viewing 3D plots.

        stacklevel: (Advanced) Choose the level of the stack that you want to retrieve plot labels at. Higher
        integers will get you higher (i.e., more end-user-facing) in the stack. Same behaviour as the `stacklevel`
        argument in warnings.warn().

    Returns: None (in-place)

    """
    arg_values = args
    n_dimensions = len(arg_values)  # dimensionality

    ##### This section serves to try to retrieve appropriate axis labels for the plot.
    ### First, set some defaults.
    arg_names = []
    if n_dimensions >= 1:
        arg_names += ["x"]
        if n_dimensions >= 2:
            arg_names += ["y"]
            if n_dimensions >= 3:
                arg_names += ["z"]
                if n_dimensions >= 4:
                    arg_names += [
                        f"Dim. {i}"
                        for i in range(4, n_dimensions + 1)
                    ]
    title = "QuickPlot"
    try:
        ### This is some interesting and tricky code here: retrieves the source code of where qp() was called, as a string.
        caller_source_code = inspect_tools.get_caller_source_code(stacklevel=stacklevel + 1)
        try:
            parsed_arg_names = inspect_tools.get_function_argument_names_from_source_code(caller_source_code)
            title = "QuickPlot: " + " vs. ".join(parsed_arg_names)
            if len(parsed_arg_names) == n_dimensions:
                arg_names = parsed_arg_names
            else:
                warnings.warn(
                    f"Couldn't parse QuickPlot call signature (dimension mismatch):\n\n{caller_source_code}",
                    stacklevel=2,
                )
        except ValueError as e:
            warnings.warn(
                f"Couldn't parse QuickPlot call signature (invalid source code):\n\n{caller_source_code}",
                stacklevel=2,
            )
    except FileNotFoundError:
        warnings.warn(
            f"Couldn't parse QuickPlot call signature (missing filepath).",
            stacklevel=2,
        )

    ##### Do the plotting:

    if backend == "plotly":
        import plotly.express as px
        import plotly.graph_objects as go

        mode = "markers+lines"
        marker_dict = dict(
            size=5 if n_dimensions != 3 else 2,
            line=dict(
                width=0
            )
        )

        if n_dimensions == 1:
            fig = go.Figure(
                data=go.Scatter(
                    y=arg_values[0],
                    mode=mode,
                    marker=marker_dict
                )
            )
            fig.update_layout(
                title=title,
                xaxis_title="Array index #",
                yaxis_title=arg_names[0]
            )
        elif n_dimensions == 2:
            fig = go.Figure(
                data=go.Scatter(
                    x=arg_values[0],
                    y=arg_values[1],
                    mode=mode,
                    marker=marker_dict
                )
            )
            fig.update_layout(
                title=title,
                xaxis_title=arg_names[0],
                yaxis_title=arg_names[1]
            )
        elif n_dimensions == 3:
            fig = go.Figure(
                data=go.Scatter3d(
                    x=arg_values[0],
                    y=arg_values[1],
                    z=arg_values[2],
                    mode=mode,
                    marker=marker_dict
                ),
            )
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title=arg_names[0],
                    yaxis_title=arg_names[1],
                    zaxis_title=arg_names[2],
                )
            )
        else:
            raise ValueError("Too many inputs to plot!")
        # fig.data[0].update(mode='markers+lines')
        if orthographic:
            fig.layout.scene.camera.projection.type = "orthographic"
        if show:
            fig.show(
                renderer=plotly_renderer
            )
    else:
        raise ValueError("Bad value of `backend`!")


if __name__ == '__main__':
    import aerosandbox.numpy as np

    x = np.linspace(0, 10, 100)
    y = x ** 2
    z = np.sin(y)
    qp(x)
    qp(x, y)
    qp(x, y, z)
