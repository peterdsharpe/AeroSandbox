# Automatic version control
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from .aerodynamics import *
from .geometry import *
from .performance import *
from .casadi_helpers import *
from .library import *

