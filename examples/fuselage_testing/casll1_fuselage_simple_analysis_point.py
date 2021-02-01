import copy

from aerosandbox import *

opti = cas.Opti()  # Initialize an analysis/optimization environment


# Here's how you can make design variables:
def variable(init_val, lb=None, ub=None):
    """
    Initialize a scalar design variable.
    :param init_val: Initial guess
    :param lb: Optional lower bound
    :param ub: Optional upper bound
    :return: The variable.
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
    Initialize a scalar variable that is fixed at the value you specify. Useful, as the optimization environment will start
    creating a computational graph for all downstream quantities of this, so you can later take gradients w.r.t. this.
    :param val: Value to set.
    :return: The quasi-variable.
    """
    var = opti.variable()
    opti.set_initial(var, val)
    opti.subject_to(var == val)
    return var


# Define the 2D properties of all airfoils you want to use
generic_cambered_airfoil = Airfoil(
    CL_function=lambda alpha, Re, mach, deflection,: (  # Lift coefficient function
            (alpha * np.pi / 180) * (2 * np.pi) + 0.4550
    ),
    CDp_function=lambda alpha, Re, mach, deflection: (  # Profile drag coefficient function
            (1 + (alpha / 5) ** 2) * 2 * (0.074 / Re ** 0.2)
    ),
    Cm_function=lambda alpha, Re, mach, deflection: (  # Moment coefficient function
        0
    )
)
generic_airfoil = Airfoil(
    CL_function=lambda alpha, Re, mach, deflection,: (  # Lift coefficient function
            (alpha * np.pi / 180) * (2 * np.pi)
    ),
    CDp_function=lambda alpha, Re, mach, deflection: (  # Profile drag coefficient function
            (1 + (alpha / 5) ** 2) * 2 * (0.074 / Re ** 0.2)
    ),
    Cm_function=lambda alpha, Re, mach, deflection: (  # Moment coefficient function
        0
    )
)

# Define the 3D geometry you want to analyze/optimize.
# Here, all distances are in meters and all angles are in degrees.
airplane = Airplane(
    name="Peter's Glider",
    x_ref=0,  # CG location
    y_ref=0,  # CG location
    z_ref=0,  # CG location
    fuselages=[
        Fuselage(
            name="Fuselage",
            x_le=-0.25,
            y_le=0,
            z_le=-0.1,
            symmetric=False,
            xsecs=[
                FuselageXSec(
                    x_c = 0,
                    y_c=0,
                    z_c=0,
                    radius=0
                ),
                FuselageXSec(
                    x_c = 0.2,
                    y_c=0,
                    z_c=0,
                    radius=0.1
                ),
                FuselageXSec(
                    x_c = 0.5,
                    y_c=0,
                    z_c=0,
                    radius=0.12
                ),
                FuselageXSec(
                    x_c = 0.8,
                    y_c=0,
                    z_c=0,
                    radius=0.1
                ),
                FuselageXSec(
                    x_c = 1,
                    y_c=0,
                    z_c=0.08,
                    radius=0
                ),
            ]
        )
    ],
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
                    twist_angle=0,
                    airfoil=generic_cambered_airfoil,
                    control_surface_type='asymmetric',  # Aileron
                    control_surface_deflection=0,
                ),
                WingXSec(  # Tip
                    x_le=0.08,
                    y_le=1,
                    z_le=0.1,
                    chord=0.08,
                    twist_angle=-2,
                    airfoil=generic_cambered_airfoil,
                ),
            ]
        ),
    ]
)
# airplane.draw()

airplane.set_spanwise_paneling_everywhere(8)  # Set the resolution of your analysis
ap = Casll1(  # Set up the AeroProblem
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
    opti=opti  # Pass it an optimization environment to work in
)

# Solver options
p_opts = {}
s_opts = {}
s_opts["mu_strategy"] = "adaptive"
opti.solver('ipopt', p_opts, s_opts)

# Solve
sol = opti.solve()

# Postprocess

# Create solved object
ap_sol = copy.deepcopy(ap)
ap_sol.substitute_solution(sol)

ap_sol.draw(show=True) # Generates a pretty picture!

print("CL:", ap_sol.CL)
print("CD:", ap_sol.CD)
print("CY:", ap_sol.CY)
print("Cl:", ap_sol.Cl)
print("Cm:", ap_sol.Cm)
print("Cn:", ap_sol.Cn)