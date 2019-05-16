import numpy as np


class Airplane:
    def __init__(self,
                 name="Untitled",
                 XYZle=[0, 0, 0],
                 wings=[]
                 ):
        self.name = name
        self.XYZle = np.array(XYZle)
        self.wings = wings
