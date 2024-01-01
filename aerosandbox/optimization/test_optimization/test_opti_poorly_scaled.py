import aerosandbox as asb
import aerosandbox.numpy as np
import pytest


def test_opti_poorly_scaled_constraints(constraint_jacobian_condition_number=1e10):
    # Constants
    a = 1
    b = 100

    # Set up an optimization environment
    opti = asb.Opti()

    # Define optimization variables
    x = opti.variable(init_guess=10)
    y = opti.variable(init_guess=10)

    c = np.sqrt(constraint_jacobian_condition_number)

    # Define constraints
    opti.subject_to([
        x * c <= 0.9 * c,
        y / c <= 0.9 / c
    ])

    # Define objective
    f = (a - x) ** 2 + b * (y - x ** 2) ** 2
    opti.minimize(f)

    # Optimize
    sol = opti.solve()

    # Check
    assert sol(x) == pytest.approx(0.9, abs=1e-4)
    assert sol(y) == pytest.approx(0.81, abs=1e-4)


def test_opti_poorly_scaled_objective(objective_hessian_condition_number=1e10):
    opti = asb.Opti()

    x = opti.variable(init_guess=10)
    y = opti.variable(init_guess=10)

    c = np.sqrt(objective_hessian_condition_number)

    # Define objective
    f = x ** 4 * c + y ** 4 / c
    opti.minimize(f)

    # Optimize
    sol = opti.solve()

    # Check
    assert sol(x) == pytest.approx(0, abs=1e-2)
    assert sol(y) == pytest.approx(0, abs=1e-2)
    assert sol(f) == pytest.approx(0, abs=1e-4)


if __name__ == '__main__':
    pytest.main()
