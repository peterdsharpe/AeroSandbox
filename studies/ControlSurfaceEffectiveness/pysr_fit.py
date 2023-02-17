from pysr import PySRRegressor
from get_data import S, H, Arc_lengths
import numpy as np

model = PySRRegressor(
    niterations=1000000,  # < Increase me for better results
    population_size=50,
    ncyclesperiteration=700,
    binary_operators=[
        "+",
        "-",
        "*",
        "/",
        "pow",
    ],
    unary_operators=[
        # "cos",
        "exp",
        "log",
        # "sin",
        # "tan",
        # "inv(x) = 1/x",
        # ^ Custom operator (julia syntax)
    ],
    complexity_of_operators={
        # "*"  : 1,
        # "+"  : 1,
        # "pow": 1,
        # "exp": 2,
        # "cos": 3,
        # "sin": 3,
        # "tan": 5,
    },
    # complexity_of_constants=0.5,
    complexity_of_variables=2,
    constraints={
        'pow': (-1, 5),
        # 'sin': 5,
        # 'cos': 5,
        # 'tan': 5,
    },
    maxsize=40,
    # batching=True,
    # batch_size=500,
    # warm_start=True,
    # extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    loss="loss(prediction, target, weight) = weight * log(max(prediction, 1e-6) / target) ^ 2",
    # ^ Custom loss function (julia syntax)
)

weights = np.ones_like(S)
weights[(1.5 < S) & (S < 2.5)] *= 4
weights[H<1.5] *= 4

model.fit(
    np.stack([
        S.flatten(),
        H.flatten()
    ], axis=1),
    Arc_lengths.flatten(),
    weights=weights.flatten(),
    variable_names=["s", "h"],
)
