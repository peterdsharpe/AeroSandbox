import plotly.express as px
import plotly.graph_objects as go
import dash
import numpy as np
import casadi as cas

def reflect_over_XZ_plane(input_vector):
    # Takes in a vector or an array and flips the y-coordinates.
    output_vector = input_vector
    shape = output_vector.shape
    if shape[1] == 1 and shape[0] == 3:  # Vector of 3 items
        output_vector = output_vector * cas.vertcat(1, -1, 1)
    elif shape[1] == 3:  # 2D Nx3 vector
        output_vector = cas.horzcat(output_vector[:, 0], -1 * output_vector[:, 1], output_vector[:, 2])
    # elif len(shape) == 3 and shape[2] == 3:  # 3D MxNx3 vector
    #     output_vector = output_vector * cas.array([1, -1, 1])
    else:
        raise Exception("Invalid input for reflect_over_XZ_plane!")

    return output_vector

class Figure3D:
    def __init__(self):
        self.fig = go.Figure()

        # x, y, and z give the vertices
        self.x_face = []
        self.y_face = []
        self.z_face = []

        # i, j and k give the connectivity of the vertices
        self.i_face = []
        self.j_face = []
        self.k_face = []
        self.intensity_face = []

        # xe, ye, and ze give lines or outlines to draw
        self.x_edge = []
        self.y_edge = []
        self.z_edge = []

    def add_line(self, points):
        """
        Adds a line (or series of lines) to draw.
        :param points: an iterable with an arbitrary number of items. Each item is a 3D point, represented as an iterable of length 3.
        :return: None

        E.g. add_line([(0, 0, 0), (1, 0, 0)])
        """
        for p in points:
            self.x_edge.append(float(p[0]))
            self.y_edge.append(float(p[1]))
            self.z_edge.append(float(p[2]))
        self.x_edge.append(None)
        self.y_edge.append(None)
        self.z_edge.append(None)

    def add_tri(self,
            points,
            intensity=0,
            outline=False,
    ):
        """
        Adds a triangular face to draw.
        :param points: an iterable with 3 items. Each item is a 3D point, represented as an iterable of length 3.
        :param intensity: Intensity associated with this face
        :param outline: Do you want to outline this triangle? [boolean]
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
             ):
        self.fig.add_trace(
            go.Mesh3d(
                x=self.x_face,
                y=self.y_face,
                z=self.z_face,
                i=self.i_face,
                j=self.j_face,
                k=self.k_face,
                flatshading=True,
                intensity=self.intensity_face,
                colorscale="mint"
            )
        )

        # define the trace for triangle sides
        self.fig.add_trace(
            go.Scatter3d(
                x=self.x_edge,
                y=self.y_edge,
                z=self.z_edge,
                mode='lines',
                name='',
                line=dict(color='rgb(0,0,0)', width=3))
        )

        self.fig.update_layout(
            title=title,
            scene=dict(aspectmode='data'),
        )

        if show:
            self.fig.show()

        return self.fig