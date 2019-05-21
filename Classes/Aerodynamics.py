import numpy as np
import math
import matplotlib.pyplot as plt


class Panel:
    def __init__(self,
                 vertices=np.zeros([4, 3]),  # Nx3 np array, each row is a vector
                 colocation_point=np.zeros([1, 3]),
                 normal_direction=np.zeros([1, 3]),
                 influencing_objects=[],
                 ):
        self.vertices = np.array(vertices)
        self.colocation_point = np.array(colocation_point)
        self.normal_direction = np.array(normal_direction)
        self.influencing_objects=influencing_objects

        assert (np.shape(self.vertices)[0] >= 3)
        assert (np.shape(self.vertices)[1] == 3)

    def add_ring_vortex(self):
        pass

    def calculate_influence(self, point):
        pass

    def draw(self):
        pass


#
class QuadPanel(Panel):
    pass
#
#
# class Vortex:
#
#
# class Source:
