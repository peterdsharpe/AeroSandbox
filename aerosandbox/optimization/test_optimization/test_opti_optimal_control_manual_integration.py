import aerosandbox as asb
import aerosandbox.numpy as np
import pytest

"""
This test solves the following optimal control problem, borrowed from Problem Set 1 of MIT 6.255 Optimization Methods,
Fall 2020. Professor: Bart Van Parys

----------

Consider a rocket that travels along a straight path. 

Let x, v, and a be the position, velocity, and acceleration of the rocket at time t, respectively. 

By discretizing time and by taking the time increment to be unity, we obtain an approximate discrete-time model of the form:

x[t+1] = x[t] + v[t]
v[t+1] = v[t] + a[t]

We assume that the acceleration is under our control, which is controlled by the rocket thrust. 

In a rough model, the magnitude |a[t]| of the acceleration is proportional to the rate of fuel consumption at time t.

Suppose that the rocket is initially at rest at the origin, i.e., x[0] = 0 and v[0] = 0. 

We wish the rocket to take off and land softly at distance d unit after T time units, i.e., x[T] = d and v[T] = 0. 

The total fuel consumption of the rocket, given by sum( c[t] * abs(a[t]) ) (where c are positive numbers known to us),
cannot be more than available amount of fuel f. To ensure a smooth trajectory, we want to ensure that the
acceleration of the rocket does not change too abruptly, i.e., abs(a[t+1]-a[t]) is always less than or equal to some
known value delta.

Now, we want to control the rocket in a manner to minimize the maximum thrust required, which is max(abs(a[t])), 
subject to the preceding constraints.
"""


# TODO make it second order

def test_rocket_control_problem(plot=False):
    ### Constants
    T = 100
    d = 50
    delta = 1e-3
    f = 1000
    c = np.ones(T)

    ### Optimization
    opti = asb.Opti()  # set up an optimization environment

    x = opti.variable(init_guess=np.linspace(0, d, T))  # position
    v = opti.variable(init_guess=d / T, n_vars=T)  # velocity
    a = opti.variable(init_guess=0, n_vars=T)  # acceleration
    gamma = opti.variable(init_guess=0, n_vars=T)  # instantaneous fuel consumption
    a_max = opti.variable(init_guess=0)  # maximum acceleration

    opti.subject_to([
        np.diff(x) == v[:-1],  # physics
        np.diff(v) == a[:-1],  # physics
        x[0] == 0,  # boundary condition
        v[0] == 0,  # boundary condition
        x[-1] == d,  # boundary condition
        v[-1] == 0,  # boundary condition
        gamma >= c * a,  # lower bound on instantaneous fuel consumption
        gamma >= -c * a,  # lower bound on instantaneous fuel consumption
        np.sum(gamma) <= f,  # fuel consumption limit
        np.diff(a) <= delta,  # jerk limits
        np.diff(a) >= -delta,  # jerk limits
        a_max >= a,  # lower bound on maximum acceleration
        a_max >= -a,  # lower bound on maximum acceleration
    ])

    opti.minimize(a_max)  # minimize the peak acceleration

    sol = opti.solve()  # solve

    assert sol(a_max) == pytest.approx(0.02181991952, rel=1e-3)  # solved externally with Julia JuMP

    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set(palette=sns.color_palette("husl"))
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=200)
        for i, val, lab in zip(np.arange(3), [x, v, a], ["$x$", "$v$", "$a$"]):
            plt.subplot(3, 1, i + 1)
            plt.plot(sol(val), label=lab)
            plt.xlabel(r"Time [s]")
            plt.ylabel(lab)
            plt.legend()
        plt.suptitle(r"Rocket Trajectory")
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    pytest.main()
