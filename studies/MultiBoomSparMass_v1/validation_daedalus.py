from aerosandbox.structures.legacy.beams import *
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=1)

opti = cas.Opti()
beam = TubeBeam1(
    opti=opti,
    length=34 / 2,
    max_allowable_stress=570e6 / 1.75,
    bending=True,
    torsion=False
)
lift_force = 9.81 * 103.873
beam.add_uniform_load(force=lift_force / 2)
beam.setup()

# Tip deflection constraint
opti.subject_to([
    beam.u[-1] < 2,  # Source: http://web.mit.edu/drela/Public/web/hpa/hpa_structure.pdf
    beam.u[-1] > -2  # Source: http://web.mit.edu/drela/Public/web/hpa/hpa_structure.pdf
])

objective = beam.mass / 5

penalty = (
        cas.sum1((cas.diff(cas.diff(beam.nominal_diameter)) / 0.01) ** 2) / beam.n  # soft stabilizer
)

opti.minimize(objective + penalty)

p_opts = {}
s_opts = {}
s_opts["max_iter"] = 1e6  # If you need to interrupt, just use ctrl+c
s_opts["mu_strategy"] = "adaptive"
opti.solver('ipopt', p_opts, s_opts)

try:
    sol = opti.solve()
except Exception:
    print("Warning: Failed!")
    sol = opti.debug

import copy

beam_sol = sol(beam)

print(f"Beam mass: {beam_sol.mass} kg")
print(f"Wing spar mass: {2 * beam_sol.mass} kg (Wing spar consists of two of these beams)")
beam_sol.draw_bending(show=False, for_print=True)

plt.savefig("validation_daedalus_plot.pgf")
plt.savefig("validation_daedalus_plot.pdf")
plt.show()
