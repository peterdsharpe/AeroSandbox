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


class Wing:
    def __init__(self,
                 name="Untitled",
                 XYZle=[0, 0, 0],
                 sections=[]
                 ):
        self.name = name
        self.XYZle = np.array(XYZle)
        self.sections = sections


class Wingsection:
    def __init__(self,
                 XYZle=[0, 0, 0],
                 chord=0,
                 twist=0,
                 airfoil=[]
                 ):
        self.XYZle = np.array(XYZle)
        self.chord = chord
        self.twist = twist
        self.airfoil = airfoil


class Airfoil:
    def __init__(self,
                 name="naca0012",
                 REref=1e6
                 ):
        self.name=name
        self.REref=REref
