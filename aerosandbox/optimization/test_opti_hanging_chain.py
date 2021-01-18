import aerosandbox as asb
from aerosandbox import cas
import pytest
import matplotlib.pyplot as plt

"""
Hanging Chain problem from https://web.casadi.org/blog/opti/

Next, we will visit the hanging chain problem. We consider 
N point masses, connected by springs, hung from two fixed points at (-2, 1) and (2, 1), subject to gravity.

We seek the rest position of the system, obtained by minimizing the total energy in the system.



"""


def test_opti_hanging_chain_with_callback():
    N = 40
    m = 40 / N
    D = 70 * N
    g = 9.81
    L = 1

    opti = asb.Opti()

    x = opti.variable(
        n_vars=N,
        init_guess=cas.linspace(-2, 2, N)
    )
    y = opti.variable(
        n_vars=N,
        init_guess=1
    )

    distance = cas.sqrt(  # Distance from one node to the next
        cas.diff(x) ** 2 + cas.diff(y) ** 2
    )

    potential_energy_spring = 0.5 * D * cas.sumsqr(distance - L / N)
    potential_energy_gravity = g * m * cas.sum1(y)
    potential_energy = potential_energy_spring + potential_energy_gravity

    opti.minimize(potential_energy)

    # Add end point constraints
    opti.subject_to([
        x[0] == -2,
        y[0] == 1,
        x[-1] == 2,
        y[-1] == 1
    ])

    # Add a ground constraint
    opti.subject_to(
        y >= cas.cos(0.1 * x) - 0.5
    )

    # Add a callback

    def plot(iter: int):
        plt.plot(
            opti.debug.value(x),
            opti.debug.value(y),
            ".-",
            label=f"Iter {iter}"
        )

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)

    sol = opti.solve(
        callback=plot
    )

    plt.legend()
    plt.show()

    assert sol.value(potential_energy) == pytest.approx(626.462, abs=1e-3)


if __name__ == '__main__':
    pytest.main()
