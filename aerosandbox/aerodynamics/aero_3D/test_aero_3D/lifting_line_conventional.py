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
airplane.draw() # You can use this to quickly preview your geometry!

analysis = asb.LiftingLine(  # Set up the AeroProblem
    airplane=airplane,
    op_point=asb.OperatingPoint(
        atmosphere=asb.Atmosphere(altitude=0, method="ISA"),
        velocity=10,  # m/s
        alpha=5,  # In degrees
        beta=0,  # In degrees
        p=0,  # About the body x-axis, in rad/sec
        q=0,  # About the body y-axis, in rad/sec
        r=0,  # About the body z-axis, in rad/sec
    ),
)

### Postprocess

print("CL:", analysis.CL)
print("CD:", analysis.CD)
print("CY:", analysis.CY)
print("Cl:", analysis.Cl)
print("Cm:", analysis.Cm)
print("Cn:", analysis.Cn)

# Answer from XFLR5 Viscous VLM2
# CL = 1.112
# CD = 0.057
# CL/CD = 19.499
#   Note that XFLR5 will overpredict lift for this case compared to reality, since a VLM method
#   (which is fundamentally linear) doesn't take into account any kind of viscous decambering
#   at high CL, and XFLR5 makes no adjustments for this. This will also mean that XFLR5 will
#   overpredict drag (in particular induced drag), since the circulation is overpredicted.