from aerosandbox import *
from aerosandbox.library.airfoils import e216, naca0008

# Define the 3D geometry you want to analyze/optimize.
# Here, all distances are in meters and all angles are in degrees.
airplane = Airplane(
    name="Peter's Glider",
    x_ref=0,  # CG location
    y_ref=0,  # CG location
    z_ref=0,  # CG location
    wings=[
        Wing(
            name="Main Wing",
            x_le=0,  # Coordinates of the wing's leading edge
            y_le=0,  # Coordinates of the wing's leading edge
            z_le=0,  # Coordinates of the wing's leading edge
            symmetric=True,
            xsecs=[  # The wing's cross ("X") sections
                WingXSec(  # Root
                    x_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
                    y_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
                    z_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
                    chord=0.18,
                    twist_angle=2,  # degrees
                    airfoil=e216,  # Airfoils are blended between a given XSec and the next one.
                    control_surface_type='symmetric',
                    # Flap # Control surfaces are applied between a given XSec and the next one.
                    control_surface_deflection=0,  # degrees
                ),
                WingXSec(  # Mid
                    x_le=0.01,
                    y_le=0.5,
                    z_le=0,
                    chord=0.16,
                    twist_angle=0,
                    airfoil=e216,
                    control_surface_type='asymmetric',  # Aileron
                    control_surface_deflection=0,
                ),
                WingXSec(  # Tip
                    x_le=0.08,
                    y_le=1,
                    z_le=0.1,
                    chord=0.08,
                    twist_angle=-2,
                    airfoil=e216,
                ),
            ]
        ),
        Wing(
            name="Horizontal Stabilizer",
            x_le=0.6,
            y_le=0,
            z_le=0.1,
            symmetric=True,
            xsecs=[
                WingXSec(  # root
                    x_le=0,
                    y_le=0,
                    z_le=0,
                    chord=0.1,
                    twist_angle=-10,
                    airfoil=naca0008,
                    control_surface_type='symmetric',  # Elevator
                    control_surface_deflection=0,
                ),
                WingXSec(  # tip
                    x_le=0.02,
                    y_le=0.17,
                    z_le=0,
                    chord=0.08,
                    twist_angle=-10,
                    airfoil=naca0008
                )
            ]
        ),
        Wing(
            name="Vertical Stabilizer",
            x_le=0.6,
            y_le=0,
            z_le=0.15,
            symmetric=False,
            xsecs=[
                WingXSec(
                    x_le=0,
                    y_le=0,
                    z_le=0,
                    chord=0.1,
                    twist_angle=0,
                    airfoil=naca0008,
                    control_surface_type='symmetric',  # Rudder
                    control_surface_deflection=0,
                ),
                WingXSec(
                    x_le=0.04,
                    y_le=0,
                    z_le=0.15,
                    chord=0.06,
                    twist_angle=0,
                    airfoil=naca0008
                )
            ]
        )
    ]
)
analysis = Casll1(  # Set up the AeroProblem
    airplane=airplane,
    op_point=OperatingPoint(
        density=1.225,  # kg/m^3
        viscosity=1.81e-5,  # kg/m-s
        velocity=10,  # m/s
        mach=0,  # Freestream mach number
        alpha=5,  # In degrees
        beta=0,  # In degrees
        p=0,  # About the body x-axis, in rad/sec
        q=0,  # About the body y-axis, in rad/sec
        r=0,  # About the body z-axis, in rad/sec
    )
)

# Postprocess
analysis.draw()  # Draw the solution

print("CL:", analysis.CL)
print("CD:", analysis.CD)
print("CY:", analysis.CY)
print("Cl:", analysis.Cl)
print("Cm:", analysis.Cm)
print("Cn:", analysis.Cn)
