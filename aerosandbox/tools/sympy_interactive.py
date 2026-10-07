### Facilitates using SymPy in a Jupyter Notebook

import sympy as s
from IPython.display import display


def simp(x):
    """Simplify a SymPy expression."""
    return s.simplify(x)


def show(
    rhs: s.Symbol,
    lhs: str | None = None,
    simplify=True,
):  # Display an equation
    """
    Display an equation in a Jupyter Notebook.

    Parameters
    ----------
    rhs : s.Symbol
        The right-hand side of the equation, as a SymPy expression.
    lhs : str | None
        If given, the left-hand side of the equation, as a string to be parsed by
        `sympy.symbols()`.
    simplify
        If True, simplifies the expression(s) before displaying.
    """
    if simplify:
        rhs = simp(rhs)

    if lhs is not None:
        if simplify:
            lhs = simp(lhs)
        display(s.Eq(s.symbols(lhs), rhs, evaluate=False))
    else:
        display(rhs)
