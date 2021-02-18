import copy

from aerosandbox import *
from numpy import pi

### Constants
n_sections = 6
n_panels_per_section = 4

### Airfoils
generic_airfoil = Airfoil(
    CL_function=lambda alpha, Re, mach, deflection,: (  # Lift coefficient function
            (alpha * np.pi / 180) * (2 * pi)
    ),
    CDp_function=lambda alpha, Re, mach, deflection: (  # Profile drag coefficient function
            (1 + (alpha / 5) ** 2) * 2 * (0.074 / Re ** 0.2)
    ),
    Cm_function=lambda alpha, Re, mach, deflection: (  # Moment coefficient function
        0
    )
)

# Optimization setup
opti = Opti()  # Initialize an optimization environment

n_xsecs = n_sections + 1

alpha = opti.variable(init_guess=0, freeze=True)
twists = opti.variable(
    n_vars=n_xsecs,
    init_guess=0,
    scale=1,
    # freeze=True
)
chords = opti.variable(
    n_vars=n_xsecs,
    init_guess=1,
    scale=1,
    log_transform=True
)

x_les = -chords / 4 * cas.cos(twists * pi / 180)  # og one this one
y_les = cas.linspace(0, 1, n_xsecs)
z_les = chords / 4 * cas.tan(twists * pi / 180)  # og this one

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
            xsecs=[
                WingXSec(
                    x_le=x_les[i],
                    y_le=y_les[i],
                    z_le=z_les[i],
                    chord=chords[i],  # variable(1,0.01,2),
                    twist_angle=twists[i],
                    airfoil=generic_airfoil,
                    spanwise_spacing="uniform"
                )
                for i in range(n_xsecs)
            ],
        ),
    ]
)
airplane.set_spanwise_paneling_everywhere(n_panels_per_section)
ap = Casll1(  # Set up the AeroProblem
    airplane=airplane,
    op_point=OperatingPoint(
        velocity=100,
        alpha=alpha,
        beta=0,
        p=0,
        q=0,
        r=0,
    ),
    opti=opti,
)

### Extra constraints
# Lift constraint
weight = 1000
opti.subject_to(ap.lift_force / weight >= 1)

### Objective

#
# def my_target_lift_distribution_shape(y_normalized):
#     """
#     Gives a target lift distribution shape as a function of normalized spanwise location.
#     Magnitude isn't important, only the relative shape.
#
#     Args:
#         y_normalized: Equivalent to $y / (0.5b)$, where:
#             y is the location along the wing, in dimensional units (e.g. meters).
#             b is the full-span of the airplane.
#         Ranges from 0 at root to 1 at tip.
#
#     Return: The desired relative lift coefficient at the given y_normalized location.
#         Absolute magnitude unimportant, only relative magnitude matters.
#
#     """
#     return (1 - cas.fmax(y_normalized, 0) ** 2) ** 0.5
#


# local_lift = ap.CL_locals * ap.chords / ap.airplane.c_ref
# local_y = ap.vortex_centers[:, 1]  # Location of local lift force, as a vector
# local_y_normalized = ap.vortex_centers[:, 1] / (0.5 * ap.airplane.wings[0].span())  # Normalized location of local lift
# target_lift = my_target_lift_distribution(local_y_normalized)  # Desired local lift distribution
#
# local_lift_shape = local_lift / cas.sum1(local_lift)  # Normalized version of the observed local lift
# target_lift_shape = target_lift / cas.sum1(target_lift)  # Normalized version of the target local lift
#
# lift_distribution_error = cas.sumsqr(  # 2-norm of the error vector
#     local_lift_shape - target_lift_shape
# )
#
# opti.minimize(lift_distribution_error)  # Minimize that 2-norm

opti.minimize(ap.drag_force)

"""
bell = [0.0]*48
for j in range(1,48):
    linear_modifier=.5
    bell[j] = np.real((1-((j/48)**2))**1.5)*linear_modifier

print("this"+str(ap.n_panels))
print(bell)
for j in range(1,48):
    opti.minimize(bell[j]-ap.CL_locals[j])
"""

# Solve
sol = opti.solve()

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
import seaborn as sns

sns.set(palette=sns.color_palette("tab10"))

fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
plt.plot(
    ap_sol.vortex_centers[:, 1],
    ap_sol.CL_locals * ap_sol.chords / ap_sol.airplane.c_ref,
    ".-",
    label="Optimized Loading",
    zorder=10
)
plt.annotate(
    text="Slight deviation from\ninviscid theory due\nto viscous effects",
    xy=(0.6, 0.4),
    xytext=(0.2, 0.2),
    xycoords="data",
    arrowprops={
        "color"     : "k",
        "width"     : 0.25,
        "headwidth" : 4,
        "headlength": 6,
    }
)

plt.xlabel("y [m]")
plt.ylabel(r"$CL \cdot c / c_{ref}$")
plt.title("Spanload Optimization Test")

y_e = np.linspace(0, 1, 400)
CL_e = np.sqrt(1 - y_e ** 2) * ap_sol.CL_locals[0] * ap_sol.chords[0] / ap_sol.airplane.c_ref
plt.plot(y_e, CL_e, label="Elliptical Loading (Theory)")
plt.grid(True)
plt.legend()
plt.show()
