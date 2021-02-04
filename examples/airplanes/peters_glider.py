from aerosandbox import *
from aerosandbox.library.airfoils import e216, naca0008

# Define the 3D geometry you want to analyze/optimize.
# Here, all distances are in meters and all angles are in degrees.
airplane = Airplane(
    name="Peter's Glider",
    xyz_ref=array([0, 0, 0]),  # CG location
    wings=[
        Wing(
            name="Main Wing",
            xyz_le=array([0, 0, 0]),  # Coordinates of the wing's leading edge
            symmetric=True,
            xsecs=[  # The wing's cross ("X") sections
                WingXSec(  # Root
                    xyz_le=array([0, 0, 0]),
                    # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
                    chord=0.18,
                    twist_angle=2,  # degrees
                    airfoil=e216,  # Airfoils are blended between a given XSec and the next one.
                    control_surface_is_symmetric=True,  # Flap
                    control_surface_deflection=0,  # degrees
                ),
                WingXSec(  # Mid
                    xyz_le=array([0.01, 0.5, 0]),
                    chord=0.16,
                    twist_angle=0,
                    airfoil=e216,
                    control_surface_is_symmetric=False,  # Aileron
                    control_surface_deflection=0,
                ),
                WingXSec(  # Tip
                    xyz_le=array([0.08, 1, 0.1]),
                    chord=0.08,
                    twist_angle=-2,
                    airfoil=e216,
                ),
            ]
        ),
        Wing(
            name="Horizontal Stabilizer",
            xyz_le=array([0.6, 0, 0.1]),
            symmetric=True,
            xsecs=[
                WingXSec(  # root
                    xyz_le=array([0, 0, 0]),
                    chord=0.1,
                    twist_angle=-10,
                    airfoil=naca0008,
                    control_surface_is_symmetric=True,  # Elevator
                    control_surface_deflection=0,
                ),
                WingXSec(  # tip
                    xyz_le=array([0.02, 0.17, 0]),
                    chord=0.08,
                    twist_angle=-10,
                    airfoil=naca0008
                )
            ]
        ),
        Wing(
            name="Vertical Stabilizer",
            xyz_le=array([0.6, 0, 0.15]),
            symmetric=False,
            xsecs=[
                WingXSec(
                    xyz_le=array([0, 0, 0]),
                    chord=0.1,
                    twist_angle=0,
                    airfoil=naca0008,
                    control_surface_deflection=0,
                ),
                WingXSec(
                    xyz_le=array([0.04, 0, 0.15]),
                    chord=0.06,
                    twist_angle=0,
                    airfoil=naca0008
                )
            ]
        )
    ]
)
