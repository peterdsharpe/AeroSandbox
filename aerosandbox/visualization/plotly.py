import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import aerosandbox.numpy as np

# Set the rendering to happen in browser
pio.renderers.default = "browser"


def spy(
    matrix,
    show=True,
):
    """
    Plot the sparsity pattern of a matrix.

    Parameters
    ----------
    matrix
        The matrix to plot the sparsity pattern of. [2D ndarray or CasADi array]
    show : bool
        Whether or not to show the sparsity plot.

    Returns
    -------
    go.Figure
        The figure to be plotted.
    """
    try:
        matrix = matrix.toarray()
    except Exception:
        pass
    abs_m = np.abs(matrix)
    sparsity_pattern = abs_m >= 1e-16
    matrix[sparsity_pattern] = np.log10(abs_m[sparsity_pattern] + 1e-16)
    j_index_map, i_index_map = np.meshgrid(
        np.arange(matrix.shape[1]), np.arange(matrix.shape[0])
    )

    i_index = i_index_map[sparsity_pattern]
    j_index = j_index_map[sparsity_pattern]
    val = matrix[sparsity_pattern]
    val = np.ones_like(i_index)
    fig = go.Figure(
        data=go.Heatmap(
            y=i_index,
            x=j_index,
            z=val,
            # type='heatmap',
            colorscale="RdBu",
            showscale=False,
        ),
    )
    fig.update_layout(
        plot_bgcolor="black",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            autorange="reversed",
            scaleanchor="x",
            scaleratio=1,
        ),
        width=800,
        height=800 * (matrix.shape[0] / matrix.shape[1]),
    )
    if show:
        fig.show()
    return fig


def plot_point_cloud(
    p: np.ndarray,
):
    """
    Plot an Nx3 point cloud with Plotly.

    Parameters
    ----------
    p : np.ndarray
        An Nx3 array of points to be plotted.

    Returns
    -------
    None
    """
    p = np.array(p)
    px.scatter_3d(x=p[:, 0], y=p[:, 1], z=p[:, 2]).show()
