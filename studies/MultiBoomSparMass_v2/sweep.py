"""
This script generates data used to estimate spar mass as a function of lift force (i.e. total weight) and span.
"""

### Imports
from aerosandbox.structures.beams import *
import copy

### Set up sweep variables
# n_booms = 1

# n_booms = 2
# load_location_fraction = 0.50

n_booms = 3
load_location_fraction = 0.60

res = 15
masses = np.logspace(np.log10(5), np.log10(3000), res)
spans = np.logspace(np.log10(3), np.log10(120), res)

Masses, Spans = np.meshgrid(masses, spans, indexing="ij")
Spar_Masses = np.zeros_like(Masses)

### Set up problem
opti = cas.Opti()
mass = opti.parameter()
span = opti.parameter()
beam = TubeBeam1(
    opti=opti,
    length=span / 2,
    points_per_point_load=100,
    diameter_guess=10,
    thickness=0.60e-3,
    bending=True,
    torsion=False
)
lift_force = 9.81 * mass
# load_location = opti.variable()
# opti.set_initial(load_location, 12)
# opti.subject_to([
#     load_location > 1,
#     load_location < beam.length - 1,
# ])
assert (n_booms == np.array([1,2,3])).any()
if n_booms == 2 or n_booms == 3:
    load_location = beam.length * load_location_fraction
    beam.add_point_load(location = load_location, force = -lift_force / n_booms)
beam.add_elliptical_load(force=lift_force / 2)
beam.setup()

# Constraints (in addition to stress)
opti.subject_to([
    # beam.u[-1] < 2,  # tip deflection. Source: http://web.mit.edu/drela/Public/web/hpa/hpa_structure.pdf
    # beam.u[-1] > -2  # tip deflection. Source: http://web.mit.edu/drela/Public/web/hpa/hpa_structure.pdf
    beam.du * 180 / cas.pi < 10,  # local dihedral constraint
    beam.du * 180 / cas.pi > -10,  # local anhedral constraint
    cas.diff(beam.nominal_diameter) < 0,  # manufacturability
])

# # Zero-curvature constraint (restrict to conical tube spars only)
# opti.subject_to([
#     cas.diff(cas.diff(beam.nominal_diameter)) == 0
# ])

opti.minimize(beam.mass)

p_opts = {}
s_opts = {}
s_opts["max_iter"] = 1e6  # If you need to interrupt, just use ctrl+c
# s_opts["mu_strategy"] = "adaptive"
opti.solver('ipopt', p_opts, s_opts)

### Do the sweep
for i in range(len(masses)):
    iterable = range(len(spans))
    iterable = iterable[::-1] if i % 2 != 0 else iterable
    for j in iterable:
        opti.set_value(mass, Masses[i, j])
        opti.set_value(span, Spans[i, j])

        sol = opti.solve()
        opti.set_initial(sol.value_variables())
        beam_sol = copy.deepcopy(beam).substitute_solution(sol)

        Spar_Masses[i, j] = beam_sol.mass * 2

np.save("masses", Masses)
np.save("spans", Spans)
np.save("spar_masses", Spar_Masses)

# Run a sanity check
beam_sol.draw_bending()

from fit import *
