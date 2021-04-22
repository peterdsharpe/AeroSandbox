from .common import *
from .atmosphere import *
from .aerodynamics import *
from .geometry import *
import aerosandbox.numpy as numpy
from .modeling import *
from .optimization import *
from .performance import *
from .propulsion import *
from .structures import *

__version__ = "3.0.12"


def docs():
    """
    Opens the AeroSandbox documentation.
    """
    import webbrowser
    webbrowser.open_new(
        "https://github.com/peterdsharpe/AeroSandbox/tree/master/aerosandbox"
    )  # TODO: make this redirect to a hosted ReadTheDocs, or similar.
