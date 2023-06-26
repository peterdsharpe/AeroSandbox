import aerosandbox as asb
import aerosandbox.numpy as np
import pytest


def test_quadcopter_navigation():
    opti = asb.Opti()

    N = 300
    time_final = 1
    time = np.linspace(0, time_final, N)

    left_thrust = opti.variable(init_guess=0.5, scale=1, n_vars=N, lower_bound=0, upper_bound=1)
    right_thrust = opti.variable(init_guess=0.5, scale=1, n_vars=N, lower_bound=0, upper_bound=1)

    mass = 0.1

    dyn = asb.FreeBodyDynamics(
        opti_to_add_constraints_to=opti,
        time=time,
        xe=opti.variable(init_guess=np.linspace(0, 1, N)),
        ze=opti.variable(init_guess=np.linspace(0, -1, N)),
        u=opti.variable(init_guess=0, n_vars=N),
        w=opti.variable(init_guess=0, n_vars=N),
        theta=opti.variable(init_guess=np.linspace(np.pi / 2, np.pi / 2, N)),
        q=opti.variable(init_guess=0, n_vars=N),
        X=left_thrust + right_thrust,
        M=(right_thrust - left_thrust) * 0.1 / 2,
        mass=mass,
        Iyy=0.5 * mass * 0.1 ** 2,
        g=9.81,
    )

    opti.subject_to([  # Starting state
        dyn.xe[0] == 0,
        dyn.ze[0] == 0,
        dyn.u[0] == 0,
        dyn.w[0] == 0,
        dyn.theta[0] == np.radians(90),
        dyn.q[0] == 0,
    ])

    opti.subject_to([  # Final state
        dyn.xe[-1] == 1,
        dyn.ze[-1] == -1,
        dyn.u[-1] == 0,
        dyn.w[-1] == 0,
        dyn.theta[-1] == np.radians(90),
        dyn.q[-1] == 0,
    ])

    effort = np.sum(  # The average "effort per second", where effort is integrated as follows:
        np.trapz(left_thrust ** 2 + right_thrust ** 2) * np.diff(time)
    ) / time_final

    opti.minimize(effort)

    sol = opti.solve()
    dyn = sol(dyn)

    assert sol.value(effort) == pytest.approx(0.714563, rel=0.01)

    print(sol.value(effort))


def test_quadcopter_flip():
    opti = asb.Opti()

    N = 300
    time_final = opti.variable(init_guess=1, lower_bound=0)
    time = np.linspace(0, time_final, N)

    left_thrust = opti.variable(init_guess=0.7, scale=1, n_vars=N, lower_bound=0, upper_bound=1)
    right_thrust = opti.variable(init_guess=0.6, scale=1, n_vars=N, lower_bound=0, upper_bound=1)

    mass = 0.1

    dyn = asb.FreeBodyDynamics(
        opti_to_add_constraints_to=opti,
        time=time,
        xe=opti.variable(init_guess=np.linspace(0, 1, N)),
        ze=opti.variable(init_guess=0, n_vars=N),
        u=opti.variable(init_guess=0, n_vars=N),
        w=opti.variable(init_guess=0, n_vars=N),
        theta=opti.variable(init_guess=np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, N)),
        q=opti.variable(init_guess=0, n_vars=N),
        X=left_thrust + right_thrust,
        M=(right_thrust - left_thrust) * 0.1 / 2,
        mass=mass,
        Iyy=0.5 * mass * 0.1 ** 2,
        g=9.81,
    )

    opti.subject_to([  # Starting state
        dyn.xe[0] == 0,
        dyn.ze[0] == 0,
        dyn.u[0] == 0,
        dyn.w[0] == 0,
        dyn.theta[0] == np.radians(90),
        dyn.q[0] == 0,
    ])

    opti.subject_to([  # Final state
        dyn.xe[-1] == 1,
        dyn.ze[-1] == 0,
        dyn.u[-1] == 0,
        dyn.w[-1] == 0,
        dyn.theta[-1] == np.radians(90 - 360),
        dyn.q[-1] == 0,
    ])

    opti.minimize(time_final)

    sol = opti.solve(verbose=False)
    dyn = sol(dyn)

    assert sol.value(time_final) == pytest.approx(0.824, abs=0.01)


if __name__ == '__main__':
    pytest.main()
