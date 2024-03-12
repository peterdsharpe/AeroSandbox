from pathlib import Path

_asb_root = Path(__file__).parent

from aerosandbox.common import *
from aerosandbox.optimization import *
from aerosandbox.modeling import *
from aerosandbox.geometry import *
from aerosandbox.atmosphere import *
from aerosandbox.weights import *
from aerosandbox.performance import *
from aerosandbox.dynamics import *
from aerosandbox.aerodynamics import *
from aerosandbox.propulsion import *
from aerosandbox.structures import *

__version__ = "4.2.3"


def docs():
    """
    Opens the AeroSandbox documentation.
    """
    import webbrowser
    webbrowser.open_new(
        "https://github.com/peterdsharpe/AeroSandbox/tree/master/aerosandbox"
    )  # TODO: make this redirect to a hosted ReadTheDocs, or similar.


def run_tests():
    """
    Runs all of the AeroSandbox internal unit tests on this computer.
    """
    try:
        import pytest
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Please install `pytest` (`pip install pytest`) to run AeroSandbox unit tests.")

    import matplotlib.pyplot as plt
    with plt.ion():  # Disable blocking plotting

        pytest.main([str(_asb_root)])
