import plotly.graph_objects as go
import aerosandbox.numpy as np


def reflect_over_XZ_plane(input_vector):
    """
    Take in a vector or an array and flip the y-coordinates.

    Parameters
    ----------
    input_vector
        A vector or list of vectors to flip.

    Returns
    -------
    ndarray
        Vector with flipped sign on y-coordinate.
    """
    input_vector = np.array(input_vector)
    shape = input_vector.shape
    if len(shape) == 1:
        return input_vector * np.array([1, -1, 1])
    elif len(shape) == 2:
        if not shape[1] == 3:
            raise ValueError(
                "The function expected either a 3-element vector or a Nx3 array!"
            )
        return input_vector * np.array([1, -1, 1])
    else:
        raise ValueError(
            "The function expected either a 3-element vector or a Nx3 array!"
        )


class Figure3D:
    """
    Build up a 3D Plotly figure from faces, lines, and streamlines, then draw it.
    """

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

    def add_line(
        self,
        points,
        mirror=False,
    ):
        """
        Add a line (or series of lines) to draw.

        Parameters
        ----------
        points
            An iterable with an arbitrary number of items. Each item is a 3D point, represented
            as an iterable of length 3.
        mirror : bool
            Should we also draw a version that's mirrored over the XZ plane?

        Returns
        -------
        None

        Examples
        --------
        >>> add_line([(0, 0, 0), (1, 0, 0)])
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
            self.add_line(points=reflected_points, mirror=False)

    def add_streamline(
        self,
        points,
        mirror=False,
    ):
        """
        Add a streamline (or series of streamlines) to draw.

        Parameters
        ----------
        points
            An iterable with an arbitrary number of items. Each item is a 3D point, represented
            as an iterable of length 3.
        mirror : bool
            Should we also draw a version that's mirrored over the XZ plane?

        Returns
        -------
        None

        Examples
        --------
        >>> add_streamline([(0, 0, 0), (1, 0, 0)])
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
            self.add_streamline(points=reflected_points, mirror=False)

    def add_tri(
        self,
        points,
        intensity=0,
        outline=False,
        mirror=False,
    ):
        """
        Add a triangular face to draw.

        Parameters
        ----------
        points
            An iterable with 3 items. Each item is a 3D point, represented as an iterable of
            length 3.
        intensity
            Intensity associated with this face.
        outline : bool
            Do you want to outline this triangle?
        mirror : bool
            Should we also draw a version that's mirrored over the XZ plane?

        Returns
        -------
        None

        Examples
        --------
        >>> add_tri([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
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
                mirror=False,
            )

    def add_quad(
        self,
        points,
        intensity=0,
        outline=True,
        mirror=False,
    ):
        """
        Add a quadrilateral face to draw.

        All points should be (approximately) coplanar if you want it to look right.

        Parameters
        ----------
        points
            An iterable with 4 items. Each item is a 3D point, represented as an iterable of
            length 3. Points should be given in sequential order.
        intensity
            Intensity associated with this face.
        outline : bool
            Do you want to outline this quad?
        mirror : bool
            Should we also draw a version that's mirrored over the XZ plane?

        Returns
        -------
        None

        Examples
        --------
        >>> add_quad([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
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
                mirror=False,
            )

    def draw(
        self,
        show=True,
        title="",
        colorbar_title="",
        colorscale="viridis",
    ):
        """
        Assemble the figure from all added elements, optionally show it, and return it.

        Parameters
        ----------
        show : bool
            Whether to show the figure after assembling it.
        title : str
            Title of the figure.
        colorbar_title : str
            Title of the colorbar (for face intensities).
        colorscale : str
            Colorscale to use for face intensities.

        Returns
        -------
        go.Figure
            The assembled figure.
        """
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
                showscale=colorbar_title is not None,
            ),
        )

        # Draw lines
        self.fig.add_trace(
            go.Scatter3d(
                x=self.x_line,
                y=self.y_line,
                z=self.z_line,
                mode="lines",
                name="",
                line=dict(color="rgb(0,0,0)", width=3),
                showlegend=False,
            )
        )

        # Draw streamlines
        self.fig.add_trace(
            go.Scatter3d(
                x=self.x_streamline,
                y=self.y_streamline,
                z=self.z_streamline,
                mode="lines",
                name="",
                line=dict(color="rgba(119,0,255,200)", width=1),
                showlegend=False,
            )
        )

        self.fig.update_layout(
            title=title,
            scene=dict(aspectmode="data"),
        )

        if show:
            self.fig.show()

        return self.fig


if __name__ == "__main__":
    fig = Figure3D()
