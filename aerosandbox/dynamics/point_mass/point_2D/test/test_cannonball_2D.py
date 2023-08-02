import aerosandbox as asb
import aerosandbox.numpy as np
import pytest
from scipy import integrate

u_e_0 = 100
w_e_0 = -100
speed_0 = (u_e_0 ** 2 + w_e_0 ** 2) ** 0.5
gamma_0 = np.arctan2(-w_e_0, u_e_0)

time = np.linspace(0, 10, 501)


def get_trajectory(
        parameterization: type = asb.DynamicsPointMass2DCartesian,
        gravity=True,
        drag=True,
        plot=False
):
    if parameterization is asb.DynamicsPointMass2DCartesian:
        dyn = parameterization(
            mass_props=asb.MassProperties(mass=1),
            x_e=0,
            z_e=0,
            u_e=u_e_0,
            w_e=w_e_0,
        )
    elif parameterization is asb.DynamicsPointMass2DSpeedGamma:
        dyn = parameterization(
            mass_props=asb.MassProperties(mass=1),
            x_e=0,
            z_e=0,
            speed=speed_0,
            gamma=gamma_0,
        )
    else:
        raise ValueError("Bad value of `parameterization`!")

    def derivatives(t, y):
        this_dyn = dyn.get_new_instance_with_state(y)
        if gravity:
            this_dyn.add_gravity_force()
        q = 0.5 * 1.225 * this_dyn.speed ** 2
        if drag:
            this_dyn.add_force(
                Fx=-1 * (0.1) ** 2 * q,
                axes="wind"
            )

        return this_dyn.unpack_state(this_dyn.state_derivatives())

    res = integrate.solve_ivp(
        fun=derivatives,
        t_span=(time[0], time[-1]),
        t_eval=time,
        y0=dyn.unpack_state(),
        method="LSODA",
        # vectorized=True,
        rtol=1e-9,
        atol=1e-9,
    )

    dyn = dyn.get_new_instance_with_state(res.y)

    if plot:
        import matplotlib.pyplot as plt
        import aerosandbox.tools.pretty_plots as p

        fig, ax = plt.subplots()
        p.plot_color_by_value(dyn.x_e, dyn.altitude, c=dyn.speed, colorbar=True)
        p.equal()
        p.show_plot("Trajectory", "$x_e$", "$z_e$")

    return dyn


def test_final_position_Cartesian_with_drag():
    dyn = get_trajectory(
        parameterization=asb.DynamicsPointMass2DCartesian,
        drag=True
    )
    assert dyn[-1].x_e == pytest.approx(277.5774436945314, abs=1e-2)
    assert dyn[-1].z_e == pytest.approx(3.1722727033631886, abs=1e-2)


def test_final_position_Cartesian_no_drag():
    dyn = get_trajectory(
        parameterization=asb.DynamicsPointMass2DCartesian,
        drag=False
    )
    assert dyn[-1].x_e == pytest.approx(1000, abs=1e-2)
    assert dyn[-1].z_e == pytest.approx(-509.5, abs=1e-2)


def test_final_position_SpeedGamma_with_drag():
    dyn = get_trajectory(
        parameterization=asb.DynamicsPointMass2DSpeedGamma,
        drag=True
    )
    assert dyn[-1].x_e == pytest.approx(277.5774436945314, abs=1e-2)
    assert dyn[-1].z_e == pytest.approx(3.1722727033631886, abs=1e-2)


def test_final_position_SpeedGamma_no_drag():
    dyn = get_trajectory(
        parameterization=asb.DynamicsPointMass2DSpeedGamma,
        drag=False
    )
    assert dyn[-1].x_e == pytest.approx(1000, abs=1e-2)
    assert dyn[-1].z_e == pytest.approx(-509.5, abs=1e-2)


def test_cross_compare_with_drag():
    dyn1 = get_trajectory(
        parameterization=asb.DynamicsPointMass2DCartesian,
        drag=True
    )
    dyn2 = get_trajectory(
        parameterization=asb.DynamicsPointMass2DSpeedGamma,
        drag=True
    )
    assert dyn1[-1].x_e == pytest.approx(dyn2[-1].x_e, abs=1e-6, rel=1e-6)
    assert dyn1[-1].z_e == pytest.approx(dyn2[-1].z_e, abs=1e-6, rel=1e-6)
    assert dyn1[-1].u_e == pytest.approx(dyn2[-1].u_e, abs=1e-6, rel=1e-6)
    assert dyn1[-1].w_e == pytest.approx(dyn2[-1].w_e, abs=1e-6, rel=1e-6)


def test_cross_compare_no_drag():
    dyn1 = get_trajectory(
        parameterization=asb.DynamicsPointMass2DCartesian,
        drag=False
    )
    dyn2 = get_trajectory(
        parameterization=asb.DynamicsPointMass2DSpeedGamma,
        drag=False
    )
    assert dyn1[-1].x_e == pytest.approx(dyn2[-1].x_e, abs=1e-6, rel=1e-6)
    assert dyn1[-1].z_e == pytest.approx(dyn2[-1].z_e, abs=1e-6, rel=1e-6)
    assert dyn1[-1].u_e == pytest.approx(dyn2[-1].u_e, abs=1e-6, rel=1e-6)
    assert dyn1[-1].w_e == pytest.approx(dyn2[-1].w_e, abs=1e-6, rel=1e-6)


#
if __name__ == '__main__':
    dyn = get_trajectory(
        asb.DynamicsPointMass2DSpeedGamma,
        # plot=True
    )
    # pytest.main()
