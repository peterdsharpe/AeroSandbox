import aerosandbox as asb
import aerosandbox.numpy as np
import pytest
from scipy import integrate

t_eval = np.linspace(0, 10, 1001)
u_e_0 = 100
w_e_0 = -100
speed_0 = (u_e_0 ** 2 + w_e_0 ** 2) ** 0.5
gamma_0 = np.arctan2(-w_e_0, u_e_0)
ivp_kwargs = {
    "vectorized": True,
    "rtol": 1e-9,
    "atol": 1e-9,
}

def test_trajectory_Cartesian_with_drag(plot=False):
    dyn_init = asb.DynamicsPointMass2DCartesian(
        mass_props=asb.MassProperties(mass=1),
        x_e=0,
        z_e=0,
        u_e=u_e_0,
        w_e=w_e_0,
    )

    def derivatives(t, y):
        this_dyn = dyn_init.__class__(
            dyn_init.mass_props,
            **{k: v for k, v in zip(dyn_init.state.keys(), y)}
        )
        this_dyn.add_gravity_force()
        this_dyn.add_force(
            Fx=-0.01 * this_dyn.speed ** 2,
            axes="wind"
        )

        return tuple(this_dyn.state_derivatives().values())

    res = integrate.solve_ivp(
        fun=derivatives,
        t_span=(t_eval[0], t_eval[-1]),
        t_eval=t_eval,
        y0=tuple(dyn_init.state.values()),
        **ivp_kwargs
    )

    dyn = dyn_init.__class__(
        dyn_init.mass_props,
        **{k: v for k, v in zip(dyn_init.state.keys(), res.y)}
    )

    if plot:
        import matplotlib.pyplot as plt;
        import aerosandbox.tools.pretty_plots as p

        fig, ax = plt.subplots()
        p.plot_color_by_value(dyn.x_e, dyn.altitude, c=dyn.speed, colorbar=True)
        p.equal()
        p.show_plot("Trajectory", "$x_e$", "$z_e$")


def test_trajectory_SpeedGamma_with_drag(plot=False):
    dyn_init = asb.DynamicsPointMass2DSpeedGamma(
        mass_props=asb.MassProperties(mass=1),
        x_e=0,
        z_e=0,
        speed=speed_0,
        gamma=gamma_0,
    )

    def derivatives(t, y):
        this_dyn = dyn_init.__class__(
            dyn_init.mass_props,
            **{k: v for k, v in zip(dyn_init.state.keys(), y)}
        )
        this_dyn.add_gravity_force()
        this_dyn.add_force(
            Fx=-0.01 * this_dyn.speed ** 2,
            axes="wind"
        )

        return tuple(this_dyn.state_derivatives().values())

    res = integrate.solve_ivp(
        fun=derivatives,
        t_span=(t_eval[0], t_eval[-1]),
        t_eval=t_eval,
        y0=tuple(dyn_init.state.values()),
        **ivp_kwargs
    )

    dyn = dyn_init.__class__(
        dyn_init.mass_props,
        **{k: v for k, v in zip(dyn_init.state.keys(), res.y)}
    )

    if plot:
        import matplotlib.pyplot as plt;
        import aerosandbox.tools.pretty_plots as p

        fig, ax = plt.subplots()
        p.plot_color_by_value(dyn.x_e, dyn.altitude, c=dyn.speed, colorbar=True)
        p.equal()
        p.show_plot("Trajectory", "$x_e$", "$z_e$")

def final_position_Cartesian(drag=False):
    dyn_init = asb.DynamicsPointMass2DCartesian(
        mass_props=asb.MassProperties(mass=1),
        x_e=0,
        z_e=0,
        u_e=u_e_0,
        w_e=w_e_0,
    )

    def derivatives(t, y):
        this_dyn = dyn_init.__class__(
            dyn_init.mass_props,
            **{k: v for k, v in zip(dyn_init.state.keys(), y)}
        )
        this_dyn.add_gravity_force()
        if drag:
            this_dyn.add_force(
                Fx=-0.01 * this_dyn.speed ** 2,
                axes="wind"
            )

        return tuple(this_dyn.state_derivatives().values())

    res = integrate.solve_ivp(
        fun=derivatives,
        t_span=(t_eval[0], t_eval[-1]),
        t_eval=t_eval,
        y0=tuple(dyn_init.state.values()),
        **ivp_kwargs
    )

    dyn = dyn_init.__class__(
        dyn_init.mass_props,
        **{k: v for k, v in zip(dyn_init.state.keys(), res.y)}
    )
    if drag:
        assert dyn[-1].x_e == pytest.approx(198.53465, abs=1e-2)
        assert dyn[-1].z_e == pytest.approx(44.452918, abs=1e-2)
    else:
        assert dyn[-1].x_e == pytest.approx(1000, abs=1e-2)
        assert dyn[-1].z_e == pytest.approx(-509.5, abs=1e-2)
    return dyn[-1]

def final_position_SpeedGamma(drag=False):
    dyn_init = asb.DynamicsPointMass2DSpeedGamma(
        mass_props=asb.MassProperties(mass=1),
        x_e=0,
        z_e=0,
        speed=speed_0,
        gamma=gamma_0,
    )

    def derivatives(t, y):
        this_dyn = dyn_init.__class__(
            dyn_init.mass_props,
            **{k: v for k, v in zip(dyn_init.state.keys(), y)}
        )
        this_dyn.add_gravity_force()
        if drag:
            this_dyn.add_force(
                Fx=-0.01 * this_dyn.speed ** 2,
                axes="wind"
            )

        return tuple(this_dyn.state_derivatives().values())

    res = integrate.solve_ivp(
        fun=derivatives,
        t_span=(t_eval[0], t_eval[-1]),
        t_eval=t_eval,
        y0=tuple(dyn_init.state.values()),
        **ivp_kwargs
    )

    dyn = dyn_init.__class__(
        dyn_init.mass_props,
        **{k: v for k, v in zip(dyn_init.state.keys(), res.y)}
    )
    if drag:
        assert dyn[-1].x_e == pytest.approx(198.53465, abs=1e-2)
        assert dyn[-1].z_e == pytest.approx(44.452918, abs=1e-2)
    else:
        assert dyn[-1].x_e == pytest.approx(1000, abs=1e-2)
        assert dyn[-1].z_e == pytest.approx(-509.5, abs=1e-2)
    return dyn[-1]

def test_final_position_Cartesian():
    final_position_Cartesian(drag=True)
    final_position_Cartesian(drag=False)

def test_final_position_SpeedGamma():
    final_position_SpeedGamma(drag=True)
    final_position_SpeedGamma(drag=False)


if __name__ == '__main__':
    test_trajectory_Cartesian_with_drag(False)
    test_trajectory_SpeedGamma_with_drag(False)
    test_final_position_Cartesian()
    test_final_position_SpeedGamma()
    print(final_position_Cartesian(True))
    print(final_position_SpeedGamma(True))
    print(final_position_Cartesian(False))
    print(final_position_SpeedGamma(False))
