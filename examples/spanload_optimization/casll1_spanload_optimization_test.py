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


# Airfoils
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

# Optimization setup
n = 12 + 1
panels_per_section = 4
# twists = cas.linspace(0, 0, n)
twists = cas.vertcat(*[variable(0, -10, 10) for i in range(n)])
# chords = cas.linspace(0.2, 0.2, n)
chords = cas.vertcat(*[variable(0.1, 0.001, 1) for i in range(n)])
x_les = -chords / 4 * cas.cos(twists * cas.pi / 180)
# x_les = cas.vertcat(*[variable(0, -0.2, 0.2) for i in range(n)])
y_les = cas.linspace(0, variable(1,0.1,1), n)
z_les = chords / 4 * cas.tan(twists * cas.pi / 180)
wing_xsecs = [
    WingXSec(
        x_le=x_les[i],
        y_le=y_les[i],
        z_le=z_les[i],
        chord=chords[i],  # variable(1,0.01,2),
        twist=twists[i],
        airfoil=generic_airfoil,
        spanwise_spacing="uniform"
    )
    for i in range(n)
]

airplane = Airplane(
    name="Spanload Optimization Test",
    x_ref=0,  # CG location
    y_ref=0,  # CG location
    z_ref=0,  # CG location
    fuselages=[],
    wings=[
        Wing(
            name="Main Wing",
            x_le=0,  # Coordinates of the wing's leading edge
            y_le=0,  # Coordinates of the wing's leading edge
            z_le=0,  # Coordinates of the wing's leading edge
            symmetric=True,
            xsecs=wing_xsecs,
        ),
    ]
)
airplane.set_spanwise_paneling_everywhere(panels_per_section)
ap = Casll1(  # Set up the AeroProblem
    airplane=airplane,
    op_point=OperatingPoint(
        velocity=100,
        alpha=0,
        beta=0,
        p=0,
        q=0,  # quasi_variable(0),
        r=0,
    ),
    opti=opti,
    run_setup=False
)
# Set up the VLM optimization submatrix
ap.setup(run_symmetric_if_possible=True)

### Extra constraints
# Lift constraint
weight = 1000
# opti.subject_to(-ap.force_total_inviscid_wind[2] / 1e3 >= weight / 1e3)
opti.subject_to(-ap.force_total_wind[2] / 1e3 >= weight / 1e3)

# Objective
# opti.minimize(-ap.force_total_inviscid_wind[0])
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

print("CL:", ap_sol.CL)
print("CD:", ap_sol.CD)
print("CY:", ap_sol.CY)
print("Cl:", ap_sol.Cl)
print("Cm:", ap_sol.Cm)
print("Cn:", ap_sol.Cn)

import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
sns.set(font_scale=1)


plt.figure()
plt.plot(ap_sol.vortex_centers[:, 1], ap_sol.CL_locals * ap_sol.chords, ".-", label="Optimized Loading")
plt.annotate(
    s="Slight deviation from\ninviscid theory due\nto viscous effects",
    xy=(0.6, 0.08),
    xytext=(0.2, 0.05),
    xycoords="data",
    arrowprops={
        "color"     : "k",
        "width"     : 0.25,
        "headwidth" : 4,
        "headlength": 6,
    }
)


plt.xlabel("y [m]")
plt.ylabel(r"$CL \cdot c$")
plt.title("Spanload Optimization Test")

y_e = np.linspace(0, 1, 400)
CL_e = np.sqrt(1 - y_e ** 2) * ap_sol.CL_locals[0] * ap_sol.chords[0]
plt.plot(y_e, CL_e, label="Elliptical Loading")
plt.grid(True)
plt.legend()
plt.show()
