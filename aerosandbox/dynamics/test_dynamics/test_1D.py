from aerosandbox.dynamics.dynamics import FreeBodyDynamics
import aerosandbox as asb
import aerosandbox.numpy as np

import pytest

def test_block_move_fixed_time():
    opti = asb.Opti()

    n_timesteps=300

    time = np.linspace(0, 1, n_timesteps)

    dyn = FreeBodyDynamics(
        opti = opti,
        time = time,
        xe = opti.variable(init_guess=np.linspace(0, 1, n_timesteps)),
        u=opti.variable(init_guess=1, n_vars=n_timesteps),
        X=opti.variable(init_guess=np.linspace(1, -1, n_timesteps)),
        mass=1,
    )

    opti.subject_to([
        dyn.xe[0] == 0,
        dyn.xe[-1] == 1,
        dyn.u[0] == 0,
        dyn.u[-1] == 0,
        ])

    opti.minimize(
        np.sum(np.trapz(dyn.X ** 2) * np.diff(time))
    )

    sol = opti.solve()

    dyn.substitute_solution(sol)

    assert dyn.xe[0] == pytest.approx(0)
    assert dyn.xe[-1] == pytest.approx(1)
    assert dyn.u[0] == pytest.approx(0)
    assert dyn.u[-1] == pytest.approx(0)
    assert np.max(dyn.u) == pytest.approx(1.5, abs=0.01)
    assert dyn.X[0] == pytest.approx(6, abs=0.01)
    assert dyn.X[-1] == pytest.approx(-6, abs=0.01)

def test_block_move_fixed_time():
    opti = asb.Opti()

    n_timesteps=300

    time = np.linspace(0, 1, n_timesteps)

    dyn = FreeBodyDynamics(
        opti = opti,
        time = time,
        xe = opti.variable(init_guess=np.linspace(0, 1, n_timesteps)),
        u=opti.variable(init_guess=1, n_vars=n_timesteps),
        X=opti.variable(init_guess=np.linspace(1, -1, n_timesteps)),
        mass=1,
    )

    opti.subject_to([
        dyn.xe[0] == 0,
        dyn.xe[-1] == 1,
        dyn.u[0] == 0,
        dyn.u[-1] == 0,
        ])

    opti.minimize(
        np.sum(np.trapz(dyn.X ** 2) * np.diff(time))
    )

    sol = opti.solve()

    dyn.substitute_solution(sol)

    assert dyn.xe[0] == pytest.approx(0)
    assert dyn.xe[-1] == pytest.approx(1)
    assert dyn.u[0] == pytest.approx(0)
    assert dyn.u[-1] == pytest.approx(0)
    assert dyn.X[0] == pytest.approx(6, abs=0.1)
    assert dyn.X[-1] == pytest.approx(-6, abs=0.1)

if __name__ == '__main__':
    pytest.main()