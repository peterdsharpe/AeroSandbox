"""
A set of tools used for making prettier Matplotlib plots.
"""

### General imports
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

### Local imports
from .plotting import *
from .plotting_with_uncertainty import *
from .formatting import *
from .colors import *
from .annotation import *
from .threedim import *
from .quickplot import *

sns.set_theme(
    palette=palettes["categorical"],
)

mpl.rcParams["figure.dpi"] = 200
mpl.rcParams["axes.formatter.useoffset"] = False
mpl.rcParams["contour.negative_linestyle"] = 'solid'
