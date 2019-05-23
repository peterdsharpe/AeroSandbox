import numpy as np
import math
import matplotlib.pyplot as plt
from .Plotting import *

class AeroProblem:
    def __init__(self,
                 aircraft=None,
                 panels=[]):
        self.aircraft=aircraft,
        self.panels=panels
    pass
    #TODO make this object

class Panel:
    def __init__(self,
                 vertices=None,  # Nx3 np array, each row is a vector. Just used for drawing panel
                 colocation_point=None,  # 1x3 np array
                 normal_direction=None,  # 1x3 np array, nonzero
                 influencing_objects=[],  # List of Vortexes and Sources
                 ):
        self.vertices = np.array(vertices)
        self.colocation_point = np.array(colocation_point)
        self.normal_direction = np.array(normal_direction)
        self.influencing_objects = influencing_objects

        assert (np.shape(self.vertices)[0] >= 3)
        assert (np.shape(self.vertices)[1] == 3)

    def set_colocation_point_at_centroid(self):
        centroid = np.mean(self.vertices, axis=0)
        self.colocation_point = centroid

    def add_ring_vortex(self):
        pass

    def calculate_influence(self, point):
        pass

    def draw(self,
             show=True,
             fig_to_plot_on=None,
             ax_to_plot_on=None
             ):

        # Setup
        if fig_to_plot_on == None or ax_to_plot_on == None:
            fig, ax = fig3d()
            fig.set_size_inches(12, 9)
        else:
            fig = fig_to_plot_on
            ax = ax_to_plot_on

        # Plot vertices
        if not (self.vertices == None).all():
            verts_to_draw = np.vstack((self.vertices, self.vertices[0, :]))
            x = verts_to_draw[:, 0]
            y = verts_to_draw[:, 1]
            z = verts_to_draw[:, 2]
            ax.plot(x, y, z, color='#be00cc', linestyle='--')

        # Plot colocation point
        if not (self.colocation_point == None).all():
            x = self.colocation_point[0]
            y = self.colocation_point[1]
            z = self.colocation_point[2]
            ax.scatter(x, y, z, color='#be00cc', marker='*')

        set_axes_equal(ax)
        plt.tight_layout()
        if show:
            plt.show()


class HorseshoeVortex:
    # As coded, can only have two points not at infinity (3-leg horseshoe vortex)
    # Wake assumed to trail to infinity in the x-direction.
    def __init__(self,
                 vertices=None, # 2x3 np array, left point first, then right.
                 strength=0,
                 ):
        self.vertices = np.array(vertices)
        self.strength = np.array(strength)

        assert (self.vertices.shape == (2, 3))

    def calculate_unit_influence(self, point):
        # Calculates the velocity induced at a point per unit vortex strength
        # Taken from Drela's Flight Vehicle Aerodynamics, pg. 132

        a = self.vertices[0, :] - point
        b = self.vertices[1, :] - point
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        x_hat = np.array([1, 0, 0])

        influence = 1 / (4 * np.pi) * (
                (np.cross(a, b) / (norm_a * norm_b + a @ b)) * (1 / norm_a + 1 / norm_b) +
                (np.cross(a, x_hat) / (norm_a - a @ x_hat)) / norm_a -
                (np.cross(b, x_hat) / (norm_b - b @ x_hat)) / norm_b
        )

        return influence


class Source:
    # A (3D) point source/sink.
    def __init__(self,
                 vertex=None,
                 strength=0,
                 ):
        self.vertex=np.array(vertex)
        self.strength=np.array(strength)

        assert (self.vertices.shape == (3))

    def calculate_unit_influence(self,point):
        r = self.vertices-point
        norm_r=np.linalg.norm(r)

        influence = 1/(4*np.pi*norm_r**2)

        return influence