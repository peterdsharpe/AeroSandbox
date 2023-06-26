import aerosandbox as asb
import aerosandbox.numpy as np
import pytest


def test_block_move_fixed_time():
    opti = asb.Opti()

    n_timesteps = 300

    time = np.linspace(0, 1, n_timesteps)

    dyn = asb.DynamicsPointMass1DHorizontal(
        mass_props=asb.MassProperties(mass=1),
        x_e=opti.variable(init_guess=np.linspace(0, 1, n_timesteps)),
        u_e=opti.variable(init_guess=1, n_vars=n_timesteps),
    )

    u = opti.variable(init_guess=np.linspace(1, -1, n_timesteps))

    dyn.add_force(
        Fx=u
    )

    dyn.constrain_derivatives(
        opti=opti,
        time=time
    )

    opti.subject_to([
        dyn.x_e[0] == 0,
        dyn.x_e[-1] == 1,
        dyn.u_e[0] == 0,
        dyn.u_e[-1] == 0,
    ])

    # effort = np.sum(
    #     np.trapz(dyn.X ** 2) * np.diff(time)
    # )

    effort = np.sum(  # More sophisticated integral-of-squares integration (closed form correct)
        np.diff(time) / 3 *
        (u[:-1] ** 2 + u[:-1] * u[1:] + u[1:] ** 2)
    )

    opti.minimize(effort)

    sol = opti.solve()

    dyn = sol(dyn)

    assert dyn.x_e[0] == pytest.approx(0)
    assert dyn.x_e[-1] == pytest.approx(1)
    assert dyn.u_e[0] == pytest.approx(0)
    assert dyn.u_e[-1] == pytest.approx(0)
    assert np.max(dyn.u_e) == pytest.approx(1.5, abs=0.01)
    assert sol.value(u)[0] == pytest.approx(6, abs=0.05)
    assert sol.value(u)[-1] == pytest.approx(-6, abs=0.05)


def test_block_move_minimum_time():
    opti = asb.Opti()

    n_timesteps = 300

    time = np.linspace(
        0,
        opti.variable(init_guess=1, lower_bound=0),
        n_timesteps,
    )

    dyn = asb.DynamicsPointMass1DHorizontal(
        mass_props=asb.MassProperties(mass=1),
        x_e=opti.variable(init_guess=np.linspace(0, 1, n_timesteps)),
        u_e=opti.variable(init_guess=1, n_vars=n_timesteps),
    )

    u = opti.variable(init_guess=np.linspace(1, -1, n_timesteps), lower_bound=-1, upper_bound=1)

    dyn.add_force(
        Fx=u
    )

    dyn.constrain_derivatives(
        opti=opti,
        time=time
    )

    opti.subject_to([
        dyn.x_e[0] == 0,
        dyn.x_e[-1] == 1,
        dyn.u_e[0] == 0,
        dyn.u_e[-1] == 0,
    ])

    opti.minimize(
        time[-1]
    )

    sol = opti.solve()

    dyn = sol(dyn)

    assert dyn.x_e[0] == pytest.approx(0)
    assert dyn.x_e[-1] == pytest.approx(1)
    assert dyn.u_e[0] == pytest.approx(0)
    assert dyn.u_e[-1] == pytest.approx(0)
    assert np.max(dyn.u_e) == pytest.approx(1, abs=0.01)
    assert sol.value(u)[0] == pytest.approx(1, abs=0.05)
    assert sol.value(u)[-1] == pytest.approx(-1, abs=0.05)
    assert np.mean(np.abs(sol.value(u))) == pytest.approx(1, abs=0.01)


if __name__ == '__main__':
    pytest.main()
