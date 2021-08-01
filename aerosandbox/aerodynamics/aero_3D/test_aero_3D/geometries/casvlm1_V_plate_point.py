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


generic_cambered_airfoil = Airfoil(
    CL_function=lambda alpha, Re, mach, deflection,: (  # Lift coefficient function
            (alpha * np.pi / 180) * (2 * np.pi) + 0.5102
    ),
    CDp_function=lambda alpha, Re, mach, deflection: (  # Profile drag function
            (1 + (alpha / 5) ** 2) * 2 * (0.074 / Re ** 0.2)
    ),
)
generic_airfoil = Airfoil(
    CL_function=lambda alpha, Re, mach, deflection,: (  # Lift coefficient function
            (alpha * np.pi / 180) * (2 * np.pi)
    ),
    CDp_function=lambda alpha, Re, mach, deflection: (  # Profile drag function
            (1 + (alpha / 5) ** 2) * 2 * (0.074 / Re ** 0.2)
    ),
)

airplane = Airplane(
    name="Flat Plate",
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
            chordwise_spacing='uniform',
            xsecs=[  # The wing's cross ("X") sections
                WingXSec(  # Root
                    x_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
                    y_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
                    z_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
                    chord=0.5,
                    twist_angle=0,  # degrees
                    airfoil=Airfoil("naca0012"),  # Airfoils are blended between a given XSec and the next one.
                    spanwise_spacing='uniform',
                ),
                WingXSec(  # Mid
                    x_le=0,
                    y_le=np.sqrt(2),
                    z_le=np.sqrt(2),
                    chord=0.5,
                    twist_angle=0,
                    airfoil=Airfoil("naca0012"),
                ),
            ]
        ),
    ]
)
airplane.set_paneling_everywhere(1, 20)
ap = Casvlm1(  # Set up the AeroProblem
    airplane=airplane,
    op_point=OperatingPoint(
        velocity=10,
        alpha=5,  # quasi_variable(5),
        beta=0,
        p=0,
        q=0,  # quasi_variable(0),
        r=0,
    ),
    opti=opti
)
# Set up the VLM optimization submatrix
ap.setup()

# Extra constraints
# Cmalpha constraint
# opti.subject_to(cas.gradient(ap.Cm, ap.op_point.alpha) == 0)

# Objective
# opti.minimize(-ap.CL_over_CDi)

# Solver options
p_opts = {}
s_opts = {}
s_opts["max_iter"] = 1e6  # If you need to interrupt, just use ctrl+c
s_opts["mu_strategy"] = "adaptive"
# s_opts["start_with_resto"] = "yes"
# s_opts["required_infeasibility_reduction"] = 0.1
opti.solver('ipopt', p_opts, s_opts)

# Solve
sol = opti.solve()

# Create solved object
ap_sol = copy.deepcopy(ap)
ap_sol.substitute_solution(sol)

# Postprocess

ap_sol.draw(ap_sol.vortex_strengths)

# Answer you should get: (XFLR5)
# CL = 0.230
# CDi = 0.003
