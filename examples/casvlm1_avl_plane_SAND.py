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
    name="AVL's plane.avl",
    x_ref=0.02463,  # CG location
    y_ref=0,  # CG location
    z_ref=0.2239,  # CG location
    s_ref=12,
    c_ref=1,
    b_ref=15,
    wings=[
        Wing(
            name="Main Wing",
            x_le=0,  # Coordinates of the wing's leading edge
            y_le=0,  # Coordinates of the wing's leading edge
            z_le=0,  # Coordinates of the wing's leading edge
            symmetric=True,
            chordwise_panels=1,
            xsecs=[  # The wing's cross ("X") sections
                WingXSec(  # Root
                    x_le=-0.25,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
                    y_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
                    z_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
                    chord=1,  # 0.18,
                    twist=4,  # variable(0,-10,10),  # degrees
                    airfoil=Airfoil(name="naca0012"),
                    control_surface_type='symmetric_problem',
                    # Flap # Control surfaces are applied between attrib_name given XSec and the next one.
                    control_surface_deflection=0,  # degrees
                    control_surface_hinge_point=0.75,  # as chord fraction
                    spanwise_panels=16,
                ),
                WingXSec(  # Mid
                    x_le=-0.175,
                    y_le=7.5,
                    z_le=0.5,
                    chord=0.7,  # 0.16,
                    twist=4,  # variable(0,-10,10),
                    airfoil=Airfoil(name="naca0012"),
                    control_surface_type='asymmetric',  # Aileron
                    control_surface_deflection=0,
                    control_surface_hinge_point=0.75
                ),
                # WingXSec(  # Tip
                #     x_c=0.08,#variable(0.08, 0, 0.16),
                #     y_c=1,#variable(1, 0.5, 1.25),
                #     z_c=0.1,#variable(0.1, 0, 0.2),
                #     chord=variable(0.08,0,1),#0.08,#variable(0.08, 0.01, 1),
                #     twist=0,#variable(0,-10,10),
                #     airfoil=Airfoil(name="naca4412"),
                # )
            ]
        ),
        Wing(
            name="Horizontal Stabilizer",
            x_le=6,
            y_le=0,
            z_le=0.5,
            symmetric=True,
            chordwise_panels=1,
            xsecs=[
                WingXSec(  # root
                    x_le=-0.1,
                    y_le=0,
                    z_le=0,
                    chord=0.4,
                    twist=variable(0, -60, 60),
                    airfoil=Airfoil(name="naca0012"),
                    control_surface_type='symmetric_problem',  # Elevator
                    control_surface_deflection=0,
                    control_surface_hinge_point=0.75,
                    spanwise_panels=10
                ),
                WingXSec(  # tip
                    x_le=-0.075,
                    y_le=2,
                    z_le=0,
                    chord=0.3,
                    twist=variable(0, -60, 60),
                    airfoil=Airfoil(name="naca0012")
                )
            ]
        ),
        Wing(
            name="Vertical Stabilizer",
            x_le=6,
            y_le=0,
            z_le=0.5,
            symmetric=False,
            chordwise_panels=1,
            xsecs=[
                WingXSec(
                    x_le=-0.1,
                    y_le=0,
                    z_le=0,
                    chord=0.4,
                    twist=0,
                    airfoil=Airfoil(name="naca0012"),
                    control_surface_type='symmetric_problem',  # Rudder
                    control_surface_deflection=0,
                    control_surface_hinge_point=0.75,
                    spanwise_panels=10
                ),
                WingXSec(
                    x_le=-0.075,
                    y_le=0,
                    z_le=1,
                    chord=0.3,
                    twist=0,
                    airfoil=Airfoil(name="naca0012")
                )
            ]
        )
    ]
)
# airplane.set_paneling_everywhere(6, 10)
ap = Casvlm1(  # Set up the AeroProblem
    airplane=airplane,
    op_point=OperatingPoint(
        velocity=65,
        density=0.002377,
        alpha=variable(0),
        beta=quasi_variable(0),
        p=quasi_variable(0),
        q=quasi_variable(0),
        r=quasi_variable(0),
    ),
    opti=opti
)
# Set up the VLM optimization submatrix
ap.setup()

# Extra constraints
# Trim constraint
opti.subject_to([
    ap.CL == 0.390510,
    ap.airplane.wings[1].xsecs[0].twist == ap.airplane.wings[1].xsecs[1].twist,
    ap.Cm == 0,
    #     -ap.force_total_inviscid_wind[2] == 9.81 * 0.5,
    #     # ap.CY == 0,
    #     # ap.Cl == 0,
    #     ap.Cm == 0,
    #     # ap.Cn == 0,
])

# Cmalpha constraint
# opti.subject_to(cas.gradient(ap.Cm, ap.op_point.alpha) * 180/np.pi == -1)

# Objective
# opti.minimize(-ap.force_total_inviscid_wind[0])

# Solver options
p_opts = {}
s_opts = {}
s_opts["max_iter"] = 1e6  # If you need to interrupt, just use ctrl+c
# s_opts["mu_strategy"] = "adaptive"
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
