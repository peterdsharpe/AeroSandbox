import aerosandbox as asb
import aerosandbox.numpy as np
from get_circle_airfoil import circle_airfoil as initial_guess_airfoil

opti = asb.Opti()

optimized_airfoil = asb.KulfanAirfoil(
    name="Optimized",
    lower_weights=opti.variable(
        init_guess=initial_guess_airfoil.lower_weights,
        lower_bound=-0.5,
        upper_bound=0.5,
    ),
    upper_weights=opti.variable(
        init_guess=initial_guess_airfoil.upper_weights,
        lower_bound=-0.5,
        upper_bound=0.5,
    ),
    leading_edge_weight=opti.variable(
        init_guess=initial_guess_airfoil.leading_edge_weight,
        lower_bound=-0.25,
        upper_bound=0.25,
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

aero = optimized_airfoil.get_aero_from_neuralfoil(
    alpha=alpha,
    Re=Re,
    mach=mach,
    model_size="xlarge"
)

opti.subject_to([
    aero["CL"] == 0.824,
    aero["CM"] >= -0.092,
    optimized_airfoil.lower_weights[0] < -0.1,
    optimized_airfoil.upper_weights[0] > 0.1,
    # optimized_airfoil.area() <= initial_guess_airfoil.area(),
])

opti.subject_to(
    optimized_airfoil.local_thickness() > 0
)

get_wiggliness = lambda af: sum([
    np.sum(np.diff(np.diff(array)) ** 2)
    for array in [af.lower_weights, af.upper_weights]
])

opti.subject_to(
    get_wiggliness(optimized_airfoil) < 4
)

opti.minimize(aero["CD"])

sol = opti.solve(
    max_iter=300, behavior_on_failure="return_last",
    verbose=False,
    options={
        # "ipopt.mu_strategy": "monotone"
    }
)

sol.show_infeasibilities(1e-9)

optimized_airfoil = sol(optimized_airfoil)
aero = sol(aero)
alpha = sol(alpha)

fig, ax = plt.subplots(figsize=(6, 3))
optimized_airfoil.draw()