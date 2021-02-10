import aerosandbox as asb
from aerosandbox import cas
import aerosandbox.numpy as np
import pytest

"""
These tests solve variants of the Rosenbrock problem:

----------
2-dimensional variant:
    Minimize:
        (a-x)**2 + b*(y-x**2)**2
        for a = 1, b = 100.


N-dimensional variant: 
    Minimize:
        sum(100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2) for i in [1, N-1]

"""
# Constants
a = 1
b = 100


def test_2D_rosenbrock():  # 2-dimensional rosenbrock
    opti = asb.Opti()  # set up an optimization environment

    # Define optimization variables
    x = opti.variable(init_guess=0)
    y = opti.variable(init_guess=0)

    # Define objective
    f = (a - x) ** 2 + b * (y - x ** 2) ** 2
    opti.minimize(f)

    # Optimize
    sol = opti.solve()

    for i in [x, y]:
        assert sol.value(i) == pytest.approx(1, abs=1e-4)


def test_2D_rosenbrock_circle_constrained():  # 2-dimensional rosenbrock, constrained to be within the unit circle.
    opti = asb.Opti()  # set up an optimization environment

    # Define optimization variables
    x = opti.variable(init_guess=0)
    y = opti.variable(init_guess=0)

    # Define constraints
    opti.subject_to(x ** 2 + y ** 2 <= 1)

    # Define objective
    f = (a - x) ** 2 + b * (y - x ** 2) ** 2
    opti.minimize(f)

    # Optimize
    sol = opti.solve()

    # Check
    assert sol.value(x) == pytest.approx(0.7864, abs=1e-4)
    assert sol.value(y) == pytest.approx(0.6177, abs=1e-4)
    # (Solution also given here: https://www.mathworks.com/help/optim/ug/example-nonlinear-constrained-minimization.html#brg0p3g-2 )


def test_ND_rosenbrock_constrained(N=10):  # N-dimensional rosenbrock
    # Problem is unimodal for N=2, N=3, and N>=8. Bimodal for 4<=N<=7. Global min is always a vector of ones.

    opti = asb.Opti()  # set up an optimization environment

    x = opti.variable(init_guess=1, n_vars=N)  # vector of variables

    objective = 0
    for i in range(N - 1):
        objective += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2

    opti.subject_to(x >= 0)  # Keeps design space unimodal for all N.

    opti.minimize(objective)

    sol = opti.solve()  # solve

    for i in range(N):
        assert sol.value(x[i]) == pytest.approx(1, abs=1e-4)


def test_2D_rosenbrock_frozen():
    opti = asb.Opti()  # set up an optimization environment

    # Define optimization variables
    x = opti.variable(init_guess=1)
    y = opti.variable(init_guess=0, freeze=True)

    # Define objective
    f = (a - x) ** 2 + b * (y - x ** 2) ** 2
    opti.minimize(f)

    # Optimize
    sol = opti.solve()

    assert sol.value(x) == pytest.approx(0.161, abs=1e-3)
    assert sol.value(y) == pytest.approx(0, abs=1e-3)
    assert sol.value(f) == pytest.approx(0.771, abs=1e-3)


if __name__ == '__main__':
    pytest.main()
