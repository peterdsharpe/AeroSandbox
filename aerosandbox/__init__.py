from aerosandbox.common import *
from aerosandbox.atmosphere import *
from aerosandbox.aerodynamics import *
from aerosandbox.dynamics import *
from aerosandbox.geometry import *
import aerosandbox.numpy as numpy
from aerosandbox.modeling import *
from aerosandbox.optimization import *
from aerosandbox.performance import *
from aerosandbox.propulsion import *
from aerosandbox.structures import *
from aerosandbox.weights import *

__version__ = "3.2.15"


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
    from pathlib import Path

    asb_root = Path(__file__).parent

    with plt.ion():  # Disable blocking plotting

        pytest.main([str(asb_root)])
