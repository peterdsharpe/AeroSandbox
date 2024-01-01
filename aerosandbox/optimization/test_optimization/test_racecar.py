import aerosandbox as asb
import aerosandbox.numpy as np
import pytest


def test_racecar(
        N=100,
        plot=False
):
    opti = asb.Opti()  # Optimization problem

    t_final = opti.variable(init_guess=1, lower_bound=0)
    t = np.linspace(0, t_final, N)

    x = opti.variable(init_guess=np.linspace(0, 1, N))
    v = opti.derivative_of(
        x, with_respect_to=t,
        derivative_init_guess=1,
        method="cubic"
    )
    u = opti.variable(init_guess=np.ones(N), lower_bound=0, upper_bound=1)
    opti.constrain_derivative(
        u - v,
        variable=v, with_respect_to=t,
        method="cubic"
    )

    from aerosandbox.numpy.integrate_discrete import (
        integrate_discrete_intervals,
        integrate_discrete_squared_curvature
    )

    effort = 0
    effort = 1e-6 * np.sum(
        integrate_discrete_squared_curvature(
            f=u,
            x=t
        )
    )

    opti.minimize(t_final + effort)

    opti.subject_to([
        v <= 1 - np.sin(2 * np.pi * x) / 2,
        x[0] == 0,
        v[0] == 0,
        x[-1] == 1,
    ])

    sol = opti.solve(
        behavior_on_failure="return_last"
    )
    print(f"t_final: {sol(t_final)}")
    print(f"error: {np.abs(1.9065661561917042 - sol(t_final))}")
    assert sol(t_final) == pytest.approx(1.9065661561917042, rel=1e-3)

    if plot:
        import matplotlib.pyplot as plt
        import aerosandbox.tools.pretty_plots as p

        fig, ax = plt.subplots()
        ax.plot(sol(t), sol(v), label="speed")
        ax.plot(sol(t), sol(x), label="pos")
        ax.plot(sol(t), 1 - np.sin(2 * np.pi * sol(x)) / 2, "r--", label="speed limit")
        ax.plot(sol(t), sol(u), "k", label="throttle")
        plt.ylim(0, 1.6)
        p.show_plot()


if __name__ == '__main__':

    N = 100  # number of control intervals

    opti = asb.Opti()  # Optimization problem

    t_final = opti.variable(init_guess=1, lower_bound=0)
    t = np.linspace(0, t_final, N)

    x = opti.variable(init_guess=np.linspace(0, 1, N))
    v = opti.derivative_of(
        x, with_respect_to=t,
        derivative_init_guess=1,
        method="cubic"
    )
    u = opti.variable(init_guess=np.ones(N), lower_bound=0, upper_bound=1)
    opti.constrain_derivative(
        u - v,
        variable=v, with_respect_to=t,
        method="cubic"
    )

    from aerosandbox.numpy.integrate_discrete import (
        integrate_discrete_intervals,
        integrate_discrete_squared_curvature
    )

    effort = 0
    effort = 1e-6 * np.sum(
        integrate_discrete_squared_curvature(
            f=u,
            x=t
        )
    )

    opti.minimize(t_final + effort)

    opti.subject_to([
        v <= 1 - np.sin(2 * np.pi * x) / 2,
        x[0] == 0,
        v[0] == 0,
        x[-1] == 1,
    ])

    sol = opti.solve(
        behavior_on_failure="return_last"
    )
    print(f"t_final: {sol(t_final)}")
    print(f"error: {np.abs(1.9065661561917042 - sol(t_final))}")

    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    fig, ax = plt.subplots()
    ax.plot(sol(t), sol(v), label="speed")
    ax.plot(sol(t), sol(x), label="pos")
    ax.plot(sol(t), 1 - np.sin(2 * np.pi * sol(x)) / 2, "r--", label="speed limit")
    ax.plot(sol(t), sol(u), ".-k", label="throttle")
    plt.ylim(0, 1.6)
    p.show_plot()
