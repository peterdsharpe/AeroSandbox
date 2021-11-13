import aerosandbox as asb
import aerosandbox.numpy as np
from scipy import integrate

t_eval = np.linspace(0, 10, 1001)

def test_trajectory_Cartesian_with_drag(plot=False):
    dyn_init = asb.DynamicsPointMass2DCartesian(
        mass_props=asb.MassProperties(mass=1),
        x_e=0,
        z_e=0,
        u_e=100,
        w_e=-100,
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
        vectorized=True
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
        speed=100 * np.sqrt(2),
        gamma=np.pi / 4,
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
        vectorized=True
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
        u_e=100,
        w_e=-100,
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
        vectorized=True
    )

    dyn = dyn_init.__class__(
        dyn_init.mass_props,
        **{k: v for k, v in zip(dyn_init.state.keys(), res.y)}
    )
    return dyn[-1]

def final_position_SpeedGamma(drag=False):
    dyn_init = asb.DynamicsPointMass2DSpeedGamma(
        mass_props=asb.MassProperties(mass=1),
        x_e=0,
        z_e=0,
        speed=100 * np.sqrt(2),
        gamma=np.pi / 4,
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
        vectorized=True
    )

    dyn = dyn_init.__class__(
        dyn_init.mass_props,
        **{k: v for k, v in zip(dyn_init.state.keys(), res.y)}
    )
    return dyn[-1]


if __name__ == '__main__':
    test_trajectory_Cartesian_with_drag(False)
    test_trajectory_SpeedGamma_with_drag(False)
