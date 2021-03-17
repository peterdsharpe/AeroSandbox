### Facilitates using SymPy in a Jupyter Notebook

import sympy as s
from IPython.display import display


def simp(x):
    return s.simplify(x)


def show(
        rhs: s.Symbol,
        lhs: str = None
):  # Display an equation
    if lhs is not None:
        display(
            s.Eq(
                s.symbols(lhs),
                simp(rhs),
                evaluate=False
            )
        )
    else:
        display(
            simp(rhs)
        )
