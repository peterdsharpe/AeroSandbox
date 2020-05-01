import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import casadi as cas
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
plt.ion()
sns.set(font_scale=1)

# Set the rendering to happen in browser
pio.renderers.default = "browser"

def reflect_over_XZ_plane(input_vector):
    # Takes in a vector or an array and flips the y-coordinates.
    output_vector = input_vector
    shape = output_vector.shape
    if len(shape) == 1 and shape[0] == 3:
        output_vector = output_vector * cas.vertcat(1, -1, 1)
    elif len(shape) == 2 and shape[1] == 1 and shape[0] == 3:  # Vector of 3 items
        output_vector = output_vector * cas.vertcat(1, -1, 1)
    elif len(shape) == 2 and shape[1] == 3:  # 2D Nx3 vector
        output_vector = cas.horzcat(output_vector[:, 0], -1 * output_vector[:, 1], output_vector[:, 2])
    # elif len(shape) == 3 and shape[2] == 3:  # 3D MxNx3 vector
    #     output_vector = output_vector * cas.array([1, -1, 1])
    else:
        raise Exception("Invalid input for reflect_over_XZ_plane!")

    return output_vector


class Figure3D:
    def __init__(self):
        self.fig = go.Figure()

        # Vertices of the faces
        self.x_face = []
        self.y_face = []
        self.z_face = []

        # Connectivity and color of the faces
        self.i_face = []
        self.j_face = []
        self.k_face = []
        self.intensity_face = []

        # Vertices of the lines
        self.x_line = []
        self.y_line = []
        self.z_line = []

        # Vertices of the streamlines
        self.x_streamline = []
        self.y_streamline = []
        self.z_streamline = []

    def add_line(self,
                 points,
                 mirror=False,
                 ):
        """
        Adds a line (or series of lines) to draw.
        :param points: an iterable with an arbitrary number of items. Each item is a 3D point, represented as an iterable of length 3.
        :param mirror: Should we also draw a version that's mirrored over the XZ plane? [boolean]
        :return: None

        E.g. add_line([(0, 0, 0), (1, 0, 0)])
        """
        for p in points:
            self.x_line.append(float(p[0]))
            self.y_line.append(float(p[1]))
            self.z_line.append(float(p[2]))
        self.x_line.append(None)
        self.y_line.append(None)
        self.z_line.append(None)
        if mirror:
            reflected_points = [reflect_over_XZ_plane(point) for point in points]
            self.add_line(
                points=reflected_points,
                mirror=False
            )

    def add_streamline(self,
                       points,
                       mirror=False,
                       ):
        """
        Adds a line (or series of lines) to draw.
        :param points: an iterable with an arbitrary number of items. Each item is a 3D point, represented as an iterable of length 3.
        :param mirror: Should we also draw a version that's mirrored over the XZ plane? [boolean]
        :return: None

        E.g. add_line([(0, 0, 0), (1, 0, 0)])
        """
        for p in points:
            self.x_streamline.append(float(p[0]))
            self.y_streamline.append(float(p[1]))
            self.z_streamline.append(float(p[2]))
        self.x_streamline.append(None)
        self.y_streamline.append(None)
        self.z_streamline.append(None)
        if mirror:
            reflected_points = [reflect_over_XZ_plane(point) for point in points]
            self.add_streamline(
                points=reflected_points,
                mirror=False
            )

    def add_tri(self,
                points,
                intensity=0,
                outline=False,
                mirror=False,
                ):
        """
        Adds a triangular face to draw.
        :param points: an iterable with 3 items. Each item is a 3D point, represented as an iterable of length 3.
        :param intensity: Intensity associated with this face
        :param outline: Do you want to outline this triangle? [boolean]
        :param mirror: Should we also draw a version that's mirrored over the XZ plane? [boolean]
        :return: None

        E.g. add_face([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
        """
        if not len(points) == 3:
            raise ValueError("'points' must have exactly 3 items!")
        for p in points:
            self.x_face.append(float(p[0]))
            self.y_face.append(float(p[1]))
            self.z_face.append(float(p[2]))
            self.intensity_face.append(intensity)
        indices_added = np.arange(len(self.x_face) - 3, len(self.x_face))
        self.i_face.append(indices_added[0])
        self.j_face.append(indices_added[1])
        self.k_face.append(indices_added[2])
        if outline:
            self.add_line(list(points) + [points[0]])
        if mirror:
            reflected_points = [reflect_over_XZ_plane(point) for point in points]
            self.add_tri(
                points=reflected_points,
                intensity=intensity,
                outline=outline,
                mirror=False
            )

    def add_quad(self,
                 points,
                 intensity=0,
                 outline=True,
                 mirror=False,
                 ):
        """
        Adds a quadrilateral face to draw. All points should be (approximately) coplanar if you want it to look right.
        :param points: an iterable with 4 items. Each item is a 3D point, represented as an iterable of length 3. Points should be given in sequential order.
        :param intensity: Intensity associated with this face
        :param outline: Do you want to outline this quad? [boolean]
        :param mirror: Should we also draw a version that's mirrored over the XZ plane? [boolean]
        :return: None

        E.g. add_face([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
        """
        if not len(points) == 4:
            raise ValueError("'points' must have exactly 4 items!")
        for p in points:
            self.x_face.append(float(p[0]))
            self.y_face.append(float(p[1]))
            self.z_face.append(float(p[2]))
            self.intensity_face.append(intensity)
        indices_added = np.arange(len(self.x_face) - 4, len(self.x_face))

        self.i_face.append(indices_added[0])
        self.j_face.append(indices_added[1])
        self.k_face.append(indices_added[2])

        self.i_face.append(indices_added[0])
        self.j_face.append(indices_added[2])
        self.k_face.append(indices_added[3])

        if outline:
            self.add_line(list(points) + [points[0]])
        if mirror:
            reflected_points = [reflect_over_XZ_plane(point) for point in points]
            self.add_quad(
                points=reflected_points,
                intensity=intensity,
                outline=outline,
                mirror=False
            )

    def draw(self,
             show=True,
             title="",
             colorbar_title="",
             colorscale="viridis",
             ):
        # Draw faces
        self.fig.add_trace(
            go.Mesh3d(
                x=self.x_face,
                y=self.y_face,
                z=self.z_face,
                i=self.i_face,
                j=self.j_face,
                k=self.k_face,
                flatshading=False,
                intensity=self.intensity_face,
                colorbar=dict(title=colorbar_title),
                colorscale=colorscale,
                showscale=colorbar_title is not None
            ),
        )

        # Draw lines
        self.fig.add_trace(
            go.Scatter3d(
                x=self.x_line,
                y=self.y_line,
                z=self.z_line,
                mode='lines',
                name='',
                line=dict(color='rgb(0,0,0)', width=3),
                showlegend=False,
            )
        )

        # Draw streamlines
        self.fig.add_trace(
            go.Scatter3d(
                x=self.x_streamline,
                y=self.y_streamline,
                z=self.z_streamline,
                mode='lines',
                name='',
                line=dict(color='rgba(119,0,255,200)', width=1),
                showlegend=False,
            )
        )

        self.fig.update_layout(
            title=title,
            scene=dict(aspectmode='data'),
        )

        if show:
            self.fig.show()

        return self.fig


def spy(
        matrix,
        show=True,
):
    """
    Plots the sparsity pattern of a matrix.
    :param matrix: The matrix to plot the sparsity pattern of. [2D ndarray or CasADi array]
    :param show: Whether or not to show the sparsity plot. [boolean]
    :return: The figure to be plotted [go.Figure]
    """
    try:
        matrix = matrix.toarray()
    except:
        pass
    abs_m = np.abs(matrix)
    sparsity_pattern = abs_m >= 1e-16
    matrix[sparsity_pattern] = np.log10(abs_m[sparsity_pattern] + 1e-16)
    j_index_map, i_index_map = np.meshgrid(np.arange(matrix.shape[1]), np.arange(matrix.shape[0]))

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
            colorscale='RdBu',
            showscale=False,
        ),
    )
    fig.update_layout(
        plot_bgcolor="black",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False, autorange="reversed", scaleanchor="x", scaleratio=1),
        width=800,
        height=800 * (matrix.shape[0] / matrix.shape[1]),
    )
    if show:
        fig.show()
    return fig


def contour(
        func,  # type: callable
        x_range,  # type: tuple
        y_range,  # type: tuple
        resolution=50,  # type: int
        show=True,  # type: bool
):
    """
    Makes a contour plot of a function of 2 variables. Can also plot a list of functions.
    :param func: function of form f(x,y) to plot.
    :param x_range: Range of x values to plot, expressed as a tuple (x_min, x_max)
    :param y_range: Range of y values to plot, expressed as a tuple (y_min, y_max)
    :param resolution: Resolution in x and y to plot. [int]
    :param show: Should we show the plot?
    :return:
    """
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
