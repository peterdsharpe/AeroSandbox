"""
This script generates data used to estimate spar mass as a function of lift force (i.e. total weight) and span.
"""

### Imports
from aerosandbox.structures.legacy.beams import *
import copy

### Set up problem
opti = cas.Opti()
mass = opti.parameter()
span = opti.parameter()
beam = TubeBeam1(
    opti=opti,
    length=span / 2,
    points_per_point_load=200,
    diameter_guess=10,
    bending=True,
    torsion=False
)
lift_force = 9.81 * mass

# location = opti.variable()
# opti.set_initial(location, span / 2 * (2/3) + 1)
# opti.subject_to([
#     location > 1,
#     location < span/2 - 1,
# ])
beam.add_point_load(
    # location=location,
    location=span / 2 * (2 / 3) + 1,
    force=-lift_force / 3
)

beam.add_uniform_load(force=lift_force / 2)
beam.setup()

# Tip deflection constraint
opti.subject_to([
    # beam.u[-1] < 2,  # Source: http://web.mit.edu/drela/Public/web/hpa/hpa_structure.pdf
    # beam.u[-1] > -2  # Source: http://web.mit.edu/drela/Public/web/hpa/hpa_structure.pdf
    beam.du * 180 / cas.pi < 10,
    beam.du * 180 / cas.pi > -10
])
# opti.subject_to([
#     cas.diff(cas.diff(beam.nominal_diameter)) < 0.002,
#     cas.diff(cas.diff(beam.nominal_diameter)) > -0.002,
# ])

# Manufacturability
opti.subject_to([
    cas.diff(beam.nominal_diameter) < 0
])

opti.minimize(beam.mass)

p_opts = {}
s_opts = {}
s_opts["max_iter"] = 1000  # If you need to interrupt, just use ctrl+c
# s_opts["mu_strategy"] = "adaptive"
# s_opts["watchdog_shortened_iter_trigger"] = 1
# s_opts["expect_infeasible_problem"]="yes"
# s_opts["start_with_resto"] = "yes"
# s_opts["required_infeasibility_reduction"] = 0.001
opti.solver('ipopt', p_opts, s_opts)

### Do the sweep
opti.set_value(mass, 350)
opti.set_value(span, 40)

sol = opti.solve()
opti.set_initial(sol.value_variables())
beam_sol = sol(beam)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=1)
beam_sol.draw_bending(for_print=True, show=False)
plt.savefig("C:/Users/User/Downloads/beam_example.png")
plt.show()
print(sol.value(2 * beam.mass))
