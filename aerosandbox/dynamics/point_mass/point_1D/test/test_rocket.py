import aerosandbox as asb
import aerosandbox.numpy as np
import pytest


def test_rocket():
    ### Environment
    opti = asb.Opti()

    ### Time discretization
    N = 500  # Number of discretization points
    time_final = 100  # seconds
    time = np.linspace(0, time_final, N)

    ### Constants
    mass_initial = 500e3  # Initial mass, 500 metric tons
    z_e_final = -100e3  # Final altitude, 100 km
    g = 9.81  # Gravity, m/s^2
    alpha = 1 / (300 * g)  # kg/(N*s), Inverse of specific impulse, basically - don't worry about this

    dyn = asb.DynamicsPointMass1DVertical(
        mass_props=asb.MassProperties(mass=opti.variable(init_guess=mass_initial, n_vars=N)),
        z_e=opti.variable(init_guess=np.linspace(0, z_e_final, N)),  # Altitude (negative due to Earth-axes convention)
        w_e=opti.variable(init_guess=-z_e_final / time_final, n_vars=N),  # Velocity
    )

    dyn.add_gravity_force(g=g)
    thrust = opti.variable(init_guess=g * mass_initial, n_vars=N)
    dyn.add_force(Fz=-thrust)

    dyn.constrain_derivatives(
        opti=opti,
        time=time,
    )

    ### Fuel burn
    opti.constrain_derivative(
        derivative=-alpha * thrust,
        variable=dyn.mass_props.mass,
        with_respect_to=time,
        method="trapezoidal",
    )

    ### Boundary conditions
    opti.subject_to([
        dyn.z_e[0] == 0,
        dyn.w_e[0] == 0,
        dyn.mass_props.mass[0] == mass_initial,
        dyn.z_e[-1] == z_e_final,
    ])

    ### Path constraints
    opti.subject_to([
        dyn.mass_props.mass >= 0,
        thrust >= 0
    ])

    ### Objective
    opti.minimize(-dyn.mass_props.mass[-1])  # Maximize the final mass == minimize fuel expenditure

    ### Solve
    sol = opti.solve(verbose=False)
    print(f"Solved in {sol.stats()['iter_count']} iterations.")
    dyn = sol(dyn)

    assert dyn.mass_props.mass[-1] == pytest.approx(290049.81034472014, rel=0.05)
    assert np.abs(dyn.w_e).max() == pytest.approx(1448, rel=0.05)


if __name__ == '__main__':
    test_rocket()
    pytest.main()
