import aerosandbox as asb
import aerosandbox.numpy as np
from get_circle_airfoil import circle_airfoil as iaf

# iaf = asb.KulfanAirfoil("rae2822")

opti = asb.Opti()

af = asb.KulfanAirfoil(
    name="Optimized",
    lower_weights=opti.variable(
        init_guess=iaf.lower_weights,
        scale=0.1,
        lower_bound=-0.5,
        upper_bound=0.5,
    ),
    upper_weights=opti.variable(
        init_guess=iaf.upper_weights,
        scale=0.1,
        lower_bound=-0.5,
        upper_bound=0.5,
    ),
    leading_edge_weight=opti.variable(
        init_guess=iaf.leading_edge_weight,
        scale=0.1,
        lower_bound=-0.5,
        upper_bound=0.5,
    ),
    TE_thickness=0,
)

alpha = opti.variable(
    init_guess=3,
    lower_bound=1,
    upper_bound=5
)
Re = 6.5e6
mach = 0.734

aero = af.get_aero_from_neuralfoil(
    alpha=alpha,
    Re=Re,
    mach=mach,
    model_size="xxlarge",
    n_crit=1,
)

opti.subject_to([
    aero["CL"] == 0.824,
    aero["CM"] >= -0.092,
    af.LE_radius() > 0.01,
    # optimized_airfoil.lower_weights[0] < -0.1,
    # optimized_airfoil.upper_weights[0] > 0.1,
    # aero["analysis_confidence"] > 0.9,
    # optimized_airfoil.area() <= initial_guess_airfoil.area(),
])

opti.subject_to(
    af.local_thickness(np.linspace(0, 1, 20)[1:-1]) > 0
)

get_wiggliness = lambda af: sum([
    np.sum(np.diff(np.diff(array)) ** 2)
    for array in [af.lower_weights, af.upper_weights]
])

opti.subject_to(
    get_wiggliness(af) < 1
)

opti.minimize(aero["CD"] / aero["analysis_confidence"])

sol = opti.solve(
    max_iter=30000000, behavior_on_failure="return_last",
    # verbose=False,
    options={
        "ipopt.mu_strategy": "monotone",
        # "ipopt.start_with_resto": "yes",
    }
)

# sol.show_infeasibilities(1e-3)

af = sol(af)
aero = sol(aero)
alpha = sol(alpha)

print("alpha:", alpha)
for key in ["CL", "CD", "CM", "analysis_confidence", "Top_Xtr", "Bot_Xtr", "mach_crit", "mach_dd"]:
    print(f"{key}: {aero[key]}")


import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 3))
af.draw()