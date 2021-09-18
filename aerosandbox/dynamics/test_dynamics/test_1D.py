from aerosandbox.dynamics.dynamics import FreeBodyDynamics
import aerosandbox as asb
import aerosandbox.numpy as np

import pytest


def test_block_move_fixed_time():
    opti = asb.Opti()

    n_timesteps = 300

    time = np.linspace(0, 1, n_timesteps)

    dyn = FreeBodyDynamics(
        opti=opti,
        time=time,
        xe=opti.variable(init_guess=np.linspace(0, 1, n_timesteps)),
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

    # effort = np.sum(
    #     np.trapz(dyn.X ** 2) * np.diff(time)
    # )

    effort = np.sum(  # More sophisticated integral-of-squares integration (closed form correct)
        np.diff(time) / 3 *
        (dyn.X[:-1] ** 2 + dyn.X[:-1] * dyn.X[1:] + dyn.X[1:] ** 2)
    )

    opti.minimize(effort)

    sol = opti.solve()

    dyn.substitute_solution(sol)

    assert dyn.xe[0] == pytest.approx(0)
    assert dyn.xe[-1] == pytest.approx(1)
    assert dyn.u[0] == pytest.approx(0)
    assert dyn.u[-1] == pytest.approx(0)
    assert np.max(dyn.u) == pytest.approx(1.5, abs=0.01)
    assert dyn.X[0] == pytest.approx(6, abs=0.05)
    assert dyn.X[-1] == pytest.approx(-6, abs=0.05)


def test_block_move_minimum_time():
    opti = asb.Opti()

    n_timesteps = 300

    time = np.linspace(
        0,
        opti.variable(init_guess=1, lower_bound=0),
        n_timesteps,
    )

    dyn = FreeBodyDynamics(
        opti=opti,
        time=time,
        xe=opti.variable(init_guess=np.linspace(0, 1, n_timesteps)),
        u=opti.variable(init_guess=1, n_vars=n_timesteps),
        X=opti.variable(init_guess=np.linspace(1, -1, n_timesteps), lower_bound=-1, upper_bound=1),
        mass=1,
    )

    opti.subject_to([
        dyn.xe[0] == 0,
        dyn.xe[-1] == 1,
        dyn.u[0] == 0,
        dyn.u[-1] == 0,
    ])

    opti.minimize(
        time[-1]
    )

    sol = opti.solve()

    dyn.substitute_solution(sol)

    assert dyn.xe[0] == pytest.approx(0)
    assert dyn.xe[-1] == pytest.approx(1)
    assert dyn.u[0] == pytest.approx(0)
    assert dyn.u[-1] == pytest.approx(0)
    assert np.max(dyn.u) == pytest.approx(1, abs=0.01)
    assert dyn.X[0] == pytest.approx(1, abs=0.05)
    assert dyn.X[-1] == pytest.approx(-1, abs=0.05)
    assert np.mean(np.abs(dyn.X)) == pytest.approx(1, abs=0.01)


def test_rocket_primitive():
    ### Parameters
    N = 100  # Number of discretization points
    time_final = 100  # seconds
    time = np.linspace(0, time_final, N)

    ### Constants
    mass_initial = 500e3  # Initial mass, 500 metric tons
    ze_final = -100e3  # Final position, 100 km
    g = 9.81  # Gravity, m/s^2
    alpha = 1 / (300 * g)  # kg/(N*s), Inverse of specific impulse, basically - don't worry about this

    ### Environment
    opti = asb.Opti()

    ### Variables
    dyn = FreeBodyDynamics(
        opti=opti,
        time=time,
        ze=opti.variable(init_guess=np.linspace(0, ze_final, N)),  # Earth-axis z, or "negative altitude"
        u=opti.variable(init_guess=-ze_final / time_final, n_vars=N),  # Velocity
        theta=np.pi / 2,  # Point the rocket up
        mass=opti.variable(init_guess=mass_initial, n_vars=N),  # Mass
        X=opti.variable(init_guess=g * mass_initial, n_vars=N),  # Thrust force (control vector)
        g=9.81,
    )

    ### Dynamics (implemented manually for now, we'll show you more sophisticated ways to do this in the Trajectory
    # Optimization part of the tutorial later on)
    opti.subject_to([  # Forward Euler, implemented manually for now
        np.diff(dyn.mass) == np.trapz(-alpha * dyn.X) * np.diff(time)
    ])

    ### Boundary conditions
    opti.subject_to([
        dyn.ze[0] == 0,
        dyn.ze[-1] == ze_final,
        dyn.u[0] == 0,
        dyn.mass[0] == mass_initial,
    ])

    ### Path constraints
    opti.subject_to([
        dyn.mass >= 0,
        dyn.X >= 0
    ])

    ### Objective
    opti.minimize(-dyn.mass[-1])  # Maximize the final mass == minimize fuel expenditure

    ### Solve
    sol = opti.solve()

    assert sol.value(dyn.mass[-1]) == pytest.approx(290049.81034472014, rel=0.05)
    assert sol.value(dyn.u).max() == pytest.approx(1448, rel=0.05)

def test_rocket_with_data_structures():

    ### Environment
    opti = asb.Opti()

    ### Time discretization
    N = 100  # Number of discretization points
    time_final = 100  # seconds
    time = np.linspace(0, time_final, N)

    ### Constants
    mass_initial = 500e3  # Initial mass, 500 metric tons
    ze_final = -100e3  # Final altitude, 100 km
    g = 9.81  # Gravity, m/s^2
    alpha = 1 / (300 * g)  # kg/(N*s), Inverse of specific impulse, basically - don't worry about this

    dyn = FreeBodyDynamics(
        opti=opti,
        time=time,
        ze=opti.variable(init_guess=np.linspace(0, ze_final, N)),  # Altitude (negative due to Earth-axes convention)
        u=opti.variable(init_guess=-ze_final / time_final, n_vars=N),  # Velocity
        theta=np.pi / 2,
        X=opti.variable(init_guess=g * mass_initial, n_vars=N),  # Mass
        mass=opti.variable(init_guess=mass_initial, n_vars=N),  # Control vector
        g=9.81,
    )

    opti.constrain_derivative(
        derivative=-alpha * dyn.X,
        variable=dyn.mass,
        with_respect_to=dyn.time,
    )

    ### Boundary conditions
    opti.subject_to([
        dyn.ze[0] == 0,
        dyn.u[0] == 0,
        dyn.mass[0] == mass_initial,
        dyn.ze[-1] == ze_final,
        ])

    ### Path constraints
    opti.subject_to([
        dyn.mass >= 0,
        dyn.X >= 0
    ])

    ### Objective
    opti.minimize(-dyn.mass[-1])  # Maximize the final mass == minimize fuel expenditure

    ### Solve
    sol = opti.solve(verbose=False)
    print(f"Solved in {sol.stats()['iter_count']} iterations.")

    assert sol.value(dyn.mass[-1]) == pytest.approx(290049.81034472014, rel=0.05)
    assert sol.value(dyn.u).max() == pytest.approx(1448, rel=0.05)

if __name__ == '__main__':
    pytest.main()
