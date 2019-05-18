import numpy as np
import matplotlib.pyplot as plt


class Airplane:
    def __init__(self,
                 name="Untitled",
                 XYZref=[0, 0, 0],
                 wings=[],
                 Sref=1,
                 Cref=1,
                 Bref=1
                 ):
        self.name = name
        self.XYZref = np.array(XYZref)
        self.wings = wings
        self.Sref=Sref
        self.Cref=Cref
        self.Bref=Bref

    def plotGeometry(self,
                     newfigure=true
                     ):
        self.newfigure=newfigure

        # plot bodies

        # plot wings

        # format

    def setRefDimsFromWing(self,



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
        self.name = name
        self.REref = REref
