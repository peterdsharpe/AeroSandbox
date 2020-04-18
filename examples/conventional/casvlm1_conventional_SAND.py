import copy

from aerosandbox import *

opti = cas.Opti()  # Initialize an optimization environment


def variable(init_val, lb=None, ub=None):
    """
    Initialize attrib_name scalar design variable.
    :param init_val: Initial guess
    :param lb: Optional lower bound
    :param ub: Optional upper bound
    :return: The created variable
    """
    var = opti.variable()
    opti.set_initial(var, init_val)
    if lb is not None:
        opti.subject_to(var >= lb)
    if ub is not None:
        opti.subject_to(var <= ub)
    return var


def quasi_variable(val):
    """
    Initialize attrib_name scalar design variable.
    :param init_val: Initial guess
    :param lb: Optional lower bound
    :param ub: Optional upper bound
    :return: The created variable
    """
    var = opti.variable()
    opti.set_initial(var, val)
    opti.subject_to(var == val)
    return var


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
                    chord=variable(0.18,0,1),#0.18,
                    twist=0,#variable(0,-10,10),  # degrees
                    airfoil=Airfoil(name="naca4412"),
                    control_surface_type='symmetric',
                    # Flap # Control surfaces are applied between attrib_name given XSec and the next one.
                    control_surface_deflection=0,  # degrees
                    control_surface_hinge_point=0.75  # as chord fraction
                ),
                WingXSec(  # Mid
                    x_le=0.01,
                    y_le=0.5,
                    z_le=0,
                    chord=variable(0.16,0,1),#0.16,
                    twist=0,#variable(0,-10,10),
                    airfoil=Airfoil(name="naca4412"),
                    control_surface_type='asymmetric',  # Aileron
                    control_surface_deflection=0,
                    control_surface_hinge_point=0.75
                ),
                WingXSec(  # Tip
                    x_le=0.08,#variable(0.08, 0, 0.16),
                    y_le=1,#variable(1, 0.5, 1.25),
                    z_le=0.1,#variable(0.1, 0, 0.2),
                    chord=variable(0.08,0,1),#0.08,#variable(0.08, 0.01, 1),
                    twist=0,#variable(0,-10,10),
                    airfoil=Airfoil(name="naca4412"),
                )
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
                    twist=variable(-10,-30,0),
                    airfoil=Airfoil(name="naca0012"),
                    control_surface_type='symmetric',  # Elevator
                    control_surface_deflection=0,
                    control_surface_hinge_point=0.75
                ),
                WingXSec(  # tip
                    x_le=0.02,
                    y_le=0.17,
                    z_le=0,
                    chord=0.08,
                    twist=variable(-10,-30,0),
                    airfoil=Airfoil(name="naca0012")
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
                    airfoil=Airfoil(name="naca0012"),
                    control_surface_type='symmetric',  # Rudder
                    control_surface_deflection=0,
                    control_surface_hinge_point=0.75
                ),
                WingXSec(
                    x_le=0.04,
                    y_le=0,
                    z_le=0.15,
                    chord=0.06,
                    twist=0,
                    airfoil=Airfoil(name="naca0012")
                )
            ]
        )
    ]
)
airplane.set_paneling_everywhere(6, 6)
ap = Casvlm1(  # Set up the AeroProblem
    airplane=airplane,
    op_point=OperatingPoint(
        velocity=10,
        alpha=5,
        beta=0,
        p=0,
        q=0,
        r=0,
    ),
    opti=opti
)

# Extra constraints
# Trim constraint
opti.subject_to([
    -ap.force_total_wind[2] == 9.81 * 0.5,
    # ap.CY == 0,
    # ap.Cl == 0,
    ap.Cm == 0,
    # ap.Cn == 0,
])

# Cmalpha constraint
# opti.subject_to(cas.gradient(ap.Cm, ap.op_point.alpha) * 180/np.pi == -1)

# Objective
opti.minimize(-ap.force_total_wind[0])

# Solver options
p_opts = {}
s_opts = {}
s_opts["max_iter"] = 1e6  # If you need to interrupt, just use ctrl+c
s_opts["mu_strategy"] = "adaptive"
# s_opts["start_with_resto"] = "yes"
# s_opts["required_infeasibility_reduction"] = 0.1
opti.solver('ipopt', p_opts, s_opts)

# Solve
try:
    sol = opti.solve()
except RuntimeError:
    sol = opti.debug

# Create solved object
ap_sol = copy.deepcopy(ap)
ap_sol.substitute_solution(sol)

# Postprocess

ap_sol.draw()

# Answer you should get: (XFLR5)
# CL = 0.797
# CDi = 0.017
# CL/CDi = 47.211
