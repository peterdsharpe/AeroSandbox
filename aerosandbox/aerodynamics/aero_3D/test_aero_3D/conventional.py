import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.library import airfoils

airfoil = airfoils.naca0008

### Define the 3D geometry you want to analyze/optimize.
# Here, all distances are in meters and all angles are in degrees.
airplane = asb.Airplane(
    name="Peter's Glider",
    xyz_ref=[0, 0, 0],  # CG location
    wings=[
        asb.Wing(
            name="Main Wing",
            xyz_le=[0, 0, 0],  # Coordinates of the wing's leading edge
            symmetric=True, # Should this wing be mirrored across the XZ plane?
            xsecs=[  # The wing's cross ("X") sections
                asb.WingXSec(  # Root
                    xyz_le=[0,0,0],  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
                    chord=0.18,
                    twist_angle=2,  # degrees
                    airfoil=airfoil,  # Airfoils are blended between a given XSec and the next one.
                    control_surface_is_symmetric=True, # Flap (ctrl. surfs. applied between this XSec and the next one.)
                    control_surface_deflection=0,  # degrees
                ),
                asb.WingXSec(  # Mid
                    xyz_le = [0.01, 0.5, 0],
                    chord=0.16,
                    twist_angle=0,
                    airfoil=airfoil,
                    control_surface_is_symmetric=False,  # Aileron
                    control_surface_deflection=0,
                ),
                asb.WingXSec(  # Tip
                    xyz_le=[0.08, 1, 0.1],
                    chord=0.08,
                    twist_angle=-2,
                    airfoil=airfoil,
                ),
            ]
        ),
        asb.Wing(
            name="Horizontal Stabilizer",
            xyz_le=[0.6, 0, 0.1],
            symmetric=True,
            xsecs=[
                asb.WingXSec(  # root
                    xyz_le=[0, 0, 0],
                    chord=0.1,
                    twist_angle=-10,
                    airfoil=airfoil,
                    control_surface_is_symmetric=True,  # Elevator
                    control_surface_deflection=0,
                ),
                asb.WingXSec(  # tip
                    xyz_le=[0.02, 0.17, 0],
                    chord=0.08,
                    twist_angle=-10,
                    airfoil=airfoil
                )
            ]
        ),
        asb.Wing(
            name="Vertical Stabilizer",
            xyz_le=[0.6, 0, 0.15],
            symmetric=False,
            xsecs=[
                asb.WingXSec(
                    xyz_le=[0,0,0],
                    chord=0.1,
                    twist_angle=0,
                    airfoil=airfoil,
                    control_surface_is_symmetric=True,  # Rudder
                    control_surface_deflection=0,
                ),
                asb.WingXSec(
                    xyz_le=[0.04, 0, 0.15],
                    chord=0.06,
                    twist_angle=0,
                    airfoil=airfoil
                )
            ]
        )
    ]
)