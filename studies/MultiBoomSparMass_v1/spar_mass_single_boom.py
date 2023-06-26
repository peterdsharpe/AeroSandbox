"""
This script generates data used to estimate spar mass as a function of lift force (i.e. total weight) and span.
"""

### Imports
from aerosandbox.structures.legacy.beams import *
import scipy.io as sio
import copy

### Set up sweep variables
masses = np.linspace(50, 800, 50)
spans = np.linspace(30, 90, 50)

Masses, Spans = np.meshgrid(masses, spans, indexing="ij")
Spar_Masses = np.zeros_like(Masses)

### Set up problem
opti = cas.Opti()
mass = opti.parameter()
span = opti.parameter()
beam = TubeBeam1(
    opti=opti,
    length=span / 2,
    points_per_point_load=50,
    diameter_guess=100,
    bending=True,
    torsion=False
)
lift_force = 9.81 * mass

beam.add_uniform_load(force=lift_force / 2)
beam.setup()

# Tip deflection constraint
opti.subject_to([
    # beam.u[-1] < 2,  # Source: http://web.mit.edu/drela/Public/web/hpa/hpa_structure.pdf
    # beam.u[-1] > -2  # Source: http://web.mit.edu/drela/Public/web/hpa/hpa_structure.pdf
    beam.du * 180 / cas.pi < 10,
    beam.du * 180 / cas.pi > -10
])
opti.subject_to([
    cas.diff(cas.diff(beam.nominal_diameter)) < 0.002,
    cas.diff(cas.diff(beam.nominal_diameter)) > -0.002,
])

opti.minimize(beam.mass)

p_opts = {}
s_opts = {}
s_opts["max_iter"] = 500  # If you need to interrupt, just use ctrl+c
s_opts["mu_strategy"] = "adaptive"
# s_opts["watchdog_shortened_iter_trigger"] = 1
# s_opts["expect_infeasible_problem"]="yes"
# s_opts["start_with_resto"] = "yes"
# s_opts["required_infeasibility_reduction"] = 0.001
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
        beam_sol = sol(beam)

        Spar_Masses[i, j] = beam_sol.mass

sio.savemat("data_single_boom.mat",
            {
                "Masses"     : Masses,
                "Spans"      : Spans,
                "Spar_Masses": Spar_Masses
            })
