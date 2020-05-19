import copy

from aerosandbox import *
from aerosandbox.library.airfoils import generic_airfoil, generic_cambered_airfoil

opti = cas.Opti()  # Initialize an analysis/optimization environment

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
                    twist=2,  # degrees
                    airfoil=generic_cambered_airfoil,  # Airfoils are blended between a given XSec and the next one.
                    control_surface_type='symmetric',
                    # Flap # Control surfaces are applied between a given XSec and the next one.
                    control_surface_deflection=0,  # degrees
                ),
                WingXSec(  # Mid
                    x_le=0.01,
                    y_le=0.5,
                    z_le=0,
                    chord=0.16,
                    twist=0,
                    airfoil=generic_cambered_airfoil,
                    control_surface_type='asymmetric',  # Aileron
                    control_surface_deflection=0,
                ),
                WingXSec(  # Tip
                    x_le=0.08,
                    y_le=1,
                    z_le=0.1,
                    chord=0.08,
                    twist=-2,
                    airfoil=generic_cambered_airfoil,
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
                    twist=-10,
                    airfoil=generic_airfoil,
                    control_surface_type='symmetric',  # Elevator
                    control_surface_deflection=0,
                ),
                WingXSec(  # tip
                    x_le=0.02,
                    y_le=0.17,
                    z_le=0,
                    chord=0.08,
                    twist=-10,
                    airfoil=generic_airfoil
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
                    twist=0,
                    airfoil=generic_airfoil,
                    control_surface_type='symmetric',  # Rudder
                    control_surface_deflection=0,
                ),
                WingXSec(
                    x_le=0.04,
                    y_le=0,
                    z_le=0.15,
                    chord=0.06,
                    twist=0,
                    airfoil=generic_airfoil
                )
            ]
        )
    ]
)
# airplane.set_spanwise_paneling_everywhere(30)  # Set the resolution of your analysis
ap = Buildup(  # Set up the AeroProblem
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
    ),
)

# # Solver options
# p_opts = {}
# s_opts = {}
# # s_opts["mu_strategy"] = "adaptive"
# opti.solver('ipopt', p_opts, s_opts)
#
# # Solve
# try:
#     sol = opti.solve()
# except RuntimeError:
#     sol = opti.debug
#     raise Exception("An error occurred!")
#
# # Postprocess
#
# # Create solved object
# ap_sol = copy.deepcopy(ap)
# ap_sol.substitute_solution(sol)
ap_sol = ap

# ap_sol.draw(show=True)  # Generates a pretty picture!

print("CL:", ap_sol.CL)
print("CD:", ap_sol.CD)
# print("CY:", ap_sol.CY)
# print("Cl:", ap_sol.Cl)
print("Cm:", ap_sol.Cm)


# print("Cn:", ap_sol.Cn)

# Answer you should get: (XFLR5)
# CL = 0.797
# CDi = 0.017
# CL/CDi = 47.211
