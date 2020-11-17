import aerosandbox as asb
from aerosandbox import cas
import numpy as np
import pytest

"""
This test solves the N-dimensional Rosenbrock problem:

----------

Minimize:
sum(100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2) for i in [1, N-1]

"""


def test_rosenbrock_constrained():
    opti = asb.Opti()  # set up an optimization environment

    N = 10  # Problem is unimodal for N=2, N=3, and N>=8. Bimodal for 4<=N<=7. Global min is always a vector of ones.

    x = opti.variable(N, init_guess=5)

    objective = 0
    for i in range(N - 1):
        objective += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2

    opti.subject_to(x>=0) # Keeps design space unimodal for all N.

    opti.minimize(objective)

    sol = opti.solve()  # solve

    for i in range(N):
        assert sol.value(x[i]) == pytest.approx(1, abs=1e-4)


if __name__ == '__main__':
    pytest.main()
