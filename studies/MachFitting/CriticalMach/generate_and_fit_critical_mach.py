import aerosandbox as asb
import aerosandbox.numpy as np

gamma = 1.4

Cp_crit = lambda M: 2 / (gamma * M ** 2) * (
        (
                (1 + (gamma - 1) / 2 * M ** 2)
                /
                (1 + (gamma - 1) / 2)
        ) ** (gamma / (gamma - 1))
        - 1
)

# Prandtl-Glauert correction
Cp_PG = lambda Cp0, M: Cp0 / (1 - M ** 2) ** 0.5

# Karman-Tsien correction
Cp_KT = lambda Cp0, M: Cp0 / (
        (1 - M ** 2) ** 0.5
        + M ** 2 / (1 + (1 - M ** 2) ** 0.5) * (Cp0 / 2)
)

### Laitone's rule
Cp_L = lambda Cp0, M: Cp0 / (
        (1 - M ** 2) ** 0.5
        + (M ** 2) * (1 + (gamma - 1) / 2 * M ** 2) / (1 + (1 - M ** 2) ** 0.5) * (Cp0 / 2)
)

M = np.linspace(0.001, 0.999, 500)

### First, solve with PG
opti = asb.Opti()
Cp0 = opti.variable(init_guess=-1.5, n_vars=len(M), upper_bound=0)

opti.subject_to([
    Cp_crit(M) == Cp_PG(Cp0, M),
])
sol = opti.solve()

### Then, use the PG solution as an initial guess to solve with Laitone's rule
opti = asb.Opti()
Cp0 = opti.variable(init_guess=sol(Cp0), upper_bound=0)

opti.subject_to([
    Cp_crit(M) == Cp_L(Cp0, M),
])
sol = opti.solve()

### Finalize data
Cp0 = sol(Cp0)
M_crit = sol(M)

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p
    fig, ax = plt.subplots()
    # plt.plot(sol(M), ".")
    # p.sns.displot(sol(M), bins=51)
    plt.plot(Cp0, sol(M), "-")
    plt.xlim(-6, 0)
    p.show_plot(
        title="Critical Mach Number vs. $C_{p0}$",
        xlabel="Incompressible Pressure Coefficient $C_{p0}$ [-]",
        ylabel="Critical Mach Number [-]",
    )

### Fit an explicit function to the data using PySR
from pysr import PySRRegressor

# model = PySRRegressor(
#     niterations=1000000,  # < Increase me for better results
#     population_size=50,
#     ncyclesperiteration=700,
#     binary_operators=[
#         "+",
#         "-",
#         "*",
#         "/",
#         "pow",
#     ],
#     unary_operators=[
#         # "cos",
#         "exp",
#         "log",
#         # "sin",
#         # "tan",
#         # "inv(x) = 1/x",
#         # ^ Custom operator (julia syntax)
#     ],
#     complexity_of_operators={
#         "*"  : 1,
#         "+"  : 1,
#         "pow": 2,
#         "exp": 2,
#         "log": 2,
#         # "cos": 3,
#         # "sin": 3,
#         # "tan": 5,
#     },
#     # complexity_of_constants=0.5,
#     complexity_of_variables=2,
#     constraints={
#         'pow': (-1, 5),
#         # 'sin': 5,
#         # 'cos': 5,
#         # 'tan': 5,
#     },
#     maxsize=40,
#     # batching=True,
#     # batch_size=500,
#     # warm_start=True,
#     # extra_sympy_mappings={"inv": lambda x: 1 / x},
#     # ^ Define operator for SymPy as well
#     # loss="loss(prediction, target, weight) = weight * (prediction - target) ^ 2",
#     # ^ Custom loss function (julia syntax)
# )
#
# model.fit(
#     np.stack([
#         Cp0,
#     ], axis=1),
#     M_crit,
#     variable_names=[
#         "Cp0",
#     ],
# )
