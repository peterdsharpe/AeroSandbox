import numpy as np
import math
import matplotlib.pyplot as plt
from .Plotting import *

class OperatingPoint:
    def __init__(self,
                 velocity=10,
                 alpha=0,
                 beta=0,
                 p=0,
                 q=0,
                 r=0