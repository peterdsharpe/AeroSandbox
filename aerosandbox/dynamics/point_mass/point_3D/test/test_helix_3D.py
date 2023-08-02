import aerosandbox as asb
import aerosandbox.numpy as np
import pytest
from scipy import integrate

u_e_0 = 8
v_e_0 = 0
w_e_0 = -6
speed_0 = (u_e_0 ** 2 + w_e_0 ** 2) ** 0.5
gamma_0 = np.arctan2(-w_e_0, u_e_0)
track_0 = 0

time = np.linspace(0, 2 * np.pi, 501)


def get_trajectory(
        parameterization: type = asb.DynamicsPointMass3DCartesian,
        plot=False
):
    if parameterization is asb.DynamicsPointMass3DCartesian:
        dyn = parameterization(
            mass_props=asb.MassProperties(mass=1),
            x_e=0,
            y_e=0,
            z_e=0,
            u_e=u_e_0,
            v_e=v_e_0,
            w_e=w_e_0,
        )
    elif parameterization is asb.DynamicsPointMass3DSpeedGammaTrack:
        dyn = parameterization(
            mass_props=asb.MassProperties(mass=1),
            x_e=0,
            y_e=0,
            z_e=0,
            speed=speed_0,
            gamma=gamma_0,
            track=track_0,
        )
    else:
        raise ValueError("Bad value of `parameterization`!")

    def derivatives(t, y):
        this_dyn = dyn.get_new_instance_with_state(y)
        this_dyn.add_force(
            Fy=8,
            axes="wind"
        )

        return this_dyn.unpack_state(this_dyn.state_derivatives())

    res = integrate.solve_ivp(
        fun=derivatives,
        t_span=(time[0], time[-1]),
        t_eval=time,
        y0=dyn.unpack_state(),
        method="LSODA",
        rtol=1e-9,
        atol=1e-9,
    )

    dyn = dyn.get_new_instance_with_state(res.y)

    if plot:
        import matplotlib.pyplot as plt
        import aerosandbox.tools.pretty_plots as p

        fig, ax = plt.subplots()
        p.plot_color_by_value(dyn.x_e, dyn.altitude, c=dyn.speed, colorbar=True)
        # p.equal()
        p.show_plot("Trajectory", "$x_e$", "$z_e$")

    return dyn


def test_final_position_Cartesian():
    dyn = get_trajectory(
        parameterization=asb.DynamicsPointMass3DCartesian,
    )
    assert dyn[-1].x_e == pytest.approx(0, abs=1e-6)
    assert dyn[-1].y_e == pytest.approx(0, abs=1e-6)
    assert dyn[-1].z_e == pytest.approx(2 * np.pi * -6, abs=1e-6)
    assert dyn[-1].u_e == pytest.approx(8, abs=1e-6)
    assert dyn[-1].v_e == pytest.approx(0, abs=1e-6)
    assert dyn[-1].w_e == pytest.approx(-6, abs=1e-6)


def test_final_position_SpeedGammaTrack():
    dyn = get_trajectory(
        parameterization=asb.DynamicsPointMass3DSpeedGammaTrack,
    )
    assert dyn[-1].x_e == pytest.approx(0, abs=1e-6)
    assert dyn[-1].y_e == pytest.approx(0, abs=1e-6)
    assert dyn[-1].z_e == pytest.approx(2 * np.pi * -6, abs=1e-6)
    assert dyn[-1].u_e == pytest.approx(8, abs=1e-6)
    assert dyn[-1].v_e == pytest.approx(0, abs=1e-6)
    assert dyn[-1].w_e == pytest.approx(-6, abs=1e-6)


def test_cross_compare():
    dyn1 = get_trajectory(
        parameterization=asb.DynamicsPointMass3DCartesian,
    )
    dyn2 = get_trajectory(
        parameterization=asb.DynamicsPointMass3DSpeedGammaTrack,
    )
    assert dyn1[-1].x_e == pytest.approx(dyn2[-1].x_e, abs=1e-6, rel=1e-6)
    assert dyn1[-1].y_e == pytest.approx(dyn2[-1].y_e, abs=1e-6, rel=1e-6)
    assert dyn1[-1].z_e == pytest.approx(dyn2[-1].z_e, abs=1e-6, rel=1e-6)
    assert dyn1[-1].u_e == pytest.approx(dyn2[-1].u_e, abs=1e-6, rel=1e-6)
    assert dyn1[-1].v_e == pytest.approx(dyn2[-1].v_e, abs=1e-6, rel=1e-6)
    assert dyn1[-1].w_e == pytest.approx(dyn2[-1].w_e, abs=1e-6, rel=1e-6)


if __name__ == '__main__':
    # pytest.main()
    dyn1 = get_trajectory(
        asb.DynamicsPointMass3DCartesian,
        plot=True
    )
    dyn2 = get_trajectory(
        asb.DynamicsPointMass3DSpeedGammaTrack,
        plot=True
    )
