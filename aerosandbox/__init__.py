from pathlib import Path
from aerosandbox.common import (
    AeroSandboxObject,
    ExplicitAnalysis,
    ImplicitAnalysis,
    load,
)
from aerosandbox.optimization import Opti, OptiSol
from aerosandbox.modeling import (
    FittedModel,
    InterpolatedModel,
    UnstructuredInterpolatedModel,
    black_box,
)
from aerosandbox.geometry import (
    reflect_over_XZ_plane,
    Airfoil,
    KulfanAirfoil,
    Wing,
    WingXSec,
    ControlSurface,
    Fuselage,
    FuselageXSec,
    Airplane,
    Propulsor,
)
from aerosandbox.atmosphere import Atmosphere
from aerosandbox.weights import (
    MassProperties,
    mass_properties_from_radius_of_gyration,
    mass_properties_of_ellipsoid,
    mass_properties_of_sphere,
    mass_properties_of_rectangular_prism,
    mass_properties_of_cube,
)
from aerosandbox.performance import OperatingPoint
from aerosandbox.dynamics import (
    DynamicsPointMass1DHorizontal,
    DynamicsPointMass1DVertical,
    DynamicsPointMass2DCartesian,
    DynamicsPointMass2DSpeedGamma,
    DynamicsPointMass3DCartesian,
    DynamicsPointMass3DSpeedGammaTrack,
    DynamicsRigidBody2DBody,
    DynamicsRigidBody3DBodyEuler,
)
from aerosandbox.aerodynamics import (
    AirfoilInviscid,
    XFoil,
    MSES,
    VortexLatticeMethod,
    LiftingLine,
    NonlinearLiftingLine,
    AeroBuildup,
    AVL,
)

_asb_root = Path(__file__).parent

try:
    from importlib.metadata import version

    __version__ = version("AeroSandbox")
except Exception:
    __version__ = "unknown"


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
        raise ModuleNotFoundError(
            "Please install `pytest` (`pip install pytest`) to run AeroSandbox unit tests."
        )

    import matplotlib.pyplot as plt

    with plt.ion():  # Disable blocking plotting
        pytest.main([str(_asb_root)])
