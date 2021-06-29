import aerosandbox as asb
import aerosandbox.numpy as np
import pytest


def test_warm_start():
    ### Set up an optimization problem

    opti = asb.Opti()

    x = opti.variable(init_guess=10)

    opti.minimize(
        x ** 4
    )

    ### Now, solve the optimization problem and print out how many iterations were needed to solve it.

    sol = opti.solve(verbose=False)
    n_iterations_1 = sol.stats()['iter_count']
    print(f"First run: Solved in {n_iterations_1} iterations.")

    ### Then, set the initial guess of the Opti problem to the solution that was found.
    opti.set_initial_from_sol(sol)

    ### Then, re-solve the problem and print how many iterations were now needed.
    sol = opti.solve(verbose=False)
    n_iterations_2 = sol.stats()['iter_count']
    print(f"Second run: Solved in {n_iterations_2} iterations.")

    ### Assert that fewer iterations were needed after a warm-start.
    assert n_iterations_2 < n_iterations_1


if __name__ == '__main__':
    pytest.main()
