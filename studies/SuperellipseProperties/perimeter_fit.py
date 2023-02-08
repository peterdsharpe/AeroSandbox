from pysr import PySRRegressor
from perimeter import s, arc_lengths

model = PySRRegressor(
    niterations=1000000,  # < Increase me for better results
    population_size=50,
    ncyclesperiteration=700,
    binary_operators=[
        "*",
        "/",
        "+",
        "pow",
    ],
    unary_operators=[
        # "cos",
        "exp",
        # "sin",
        # "tan",
        # "inv(x) = 1/x",
        # ^ Custom operator (julia syntax)
    ],
    complexity_of_operators={
        "*"  : 1,
        "+"  : 1,
        "pow": 1,
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
    # loss="loss(prediction, target, weight) = weight * abs(prediction - target)",
    # ^ Custom loss function (julia syntax)
)

model.fit(
    s.reshape((-1, 1)), arc_lengths,
    # weights=weights,
    variable_names=["s"]
)