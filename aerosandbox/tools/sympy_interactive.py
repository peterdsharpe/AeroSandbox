### Facilitates using SymPy in a Jupyter Notebook

import sympy as s
from IPython.display import display


def simp(x):
    return s.simplify(x)


def show(
        rhs: s.Symbol,
        lhs: str = None,
        simplify=True,
):  # Display an equation
    if simplify:
        rhs = simp(rhs)

    if lhs is not None:
        display(
            s.Eq(
                s.symbols(lhs),
                rhs,
                evaluate=False
            )
        )
    else:
        display(
            rhs
        )
