from pysr import PySRRegressor
from read_data import a, d, hf, eff
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
    maxsize=30,
    # batching=True,
    # batch_size=500,
    # warm_start=True,
    # extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    loss="loss(prediction, target, weight) = weight * (prediction - target) ^ 2",
    # ^ Custom loss function (julia syntax)
)

weights = np.ones_like(eff)
weights[hf <= 0.5] *= 4
weights[d < 10] *= 2
weights[a < 10] *= 2

weights[hf == 0] = 100
weights[hf == 1] = 100

model.fit(
    np.stack([
        a,
        # d,
        hf,
        # lr
    ], axis=1),
    eff,
    weights=weights,
    variable_names=[
        "a",
        # "d",
        "hf",
        # "lr"
    ],
)
