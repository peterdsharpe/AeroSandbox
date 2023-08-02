import aerosandbox as asb
import aerosandbox.numpy as np
import pytest

u_e_0 = 100
v_e_0 = 0
w_e_0 = -100
speed_0 = (u_e_0 ** 2 + w_e_0 ** 2) ** 0.5
gamma_0 = np.arctan2(-w_e_0, u_e_0)
track_0 = 0

time = np.linspace(0, 10, 501)
N = len(time)


def get_trajectory(
        parameterization: type = asb.DynamicsPointMass3DCartesian,
        gravity=True,
        drag=True,
        sideforce=True,
        plot=False,
        verbose=False,
):
    opti = asb.Opti()

    if parameterization is asb.DynamicsPointMass3DCartesian:
        dyn = parameterization(
            mass_props=asb.MassProperties(mass=1),
            x_e=opti.variable(np.linspace(0, 300, N)),
            y_e=opti.variable(np.linspace(0, 0, N), scale=1),
            z_e=opti.variable(np.linspace(0, 0, N), scale=100),
            u_e=opti.variable(np.linspace(100, 50, N)),
            v_e=opti.variable(np.linspace(0, 0, N), scale=1),
            w_e=opti.variable(np.linspace(-100, 50, N)),
        )
    elif parameterization is asb.DynamicsPointMass3DSpeedGammaTrack:
        dyn = parameterization(
            mass_props=asb.MassProperties(mass=1),
            x_e=opti.variable(np.linspace(0, 300, N)),
            y_e=opti.variable(np.linspace(0, 0, N), scale=1),
            z_e=opti.variable(np.linspace(0, 0, N), scale=100),
            speed=opti.variable(np.linspace(100, 50, N)),
            gamma=opti.variable(np.linspace(0, 0, N)),
            track=opti.variable(np.linspace(0, 0, N)),
        )
    else:
        raise ValueError("Bad value of `parameterization`!")

    if gravity:
        dyn.add_gravity_force()
    q = 0.5 * 1.225 * dyn.speed ** 2
    if drag:
        dyn.add_force(
            Fx=-1 * (0.1) ** 2 * q,
            axes="wind"
        )
    if sideforce:
        strouhal = 0.2
        frequency = 5  # strouhal * dyn.speed / 0.1
        dyn.add_force(
            Fy=q * 1 * (0.1) ** 2 * np.sin(2 * np.pi * frequency * time),
            axes="wind"
        )

    dyn.constrain_derivatives(
        opti=opti,
        time=time,
    )

    opti.subject_to([
        dyn.x_e[0] == 0,
        dyn.y_e[0] == 0,
        dyn.z_e[0] == 0,
        dyn.u_e[0] == 100,
        dyn.v_e[0] == 0,
        dyn.w_e[0] == -100,
    ])

    sol = opti.solve(verbose=verbose)

    dyn = sol(dyn)

    if plot:
        import matplotlib.pyplot as plt
        import aerosandbox.tools.pretty_plots as p

        fig, ax = plt.subplots()
        p.plot_color_by_value(dyn.x_e, dyn.altitude, c=dyn.speed, colorbar=True)
        p.equal()
        p.show_plot("Trajectory", "$x_e$", "$z_e$")

    return dyn


def test_final_position_Cartesian_with_sideforce():
    dyn = get_trajectory(
        parameterization=asb.DynamicsPointMass3DCartesian,
        drag=True,
        sideforce=True
    )
    assert dyn[-1].x_e == pytest.approx(277.3463197415092, abs=1e-1)
    assert dyn[-1].y_e == pytest.approx(10.791223276048788, abs=1)
    assert dyn[-1].z_e == pytest.approx(3.1722726974414805, abs=1e-1)


def test_final_position_Cartesian_with_drag():
    dyn = get_trajectory(
        parameterization=asb.DynamicsPointMass3DCartesian,
        drag=True,
        sideforce=False
    )
    assert dyn[-1].x_e == pytest.approx(277.5774436945314, abs=1e-2)
    assert dyn[-1].z_e == pytest.approx(3.1722727033631886, abs=1e-2)


def test_final_position_Cartesian_no_drag():
    dyn = get_trajectory(
        parameterization=asb.DynamicsPointMass3DCartesian,
        drag=False,
        sideforce=False
    )
    assert dyn[-1].x_e == pytest.approx(1000, abs=1e-2)
    assert dyn[-1].z_e == pytest.approx(-509.5, abs=1e-2)


def test_final_position_SpeedGammaTrack_with_sideforce():
    dyn = get_trajectory(
        parameterization=asb.DynamicsPointMass3DSpeedGammaTrack,
        drag=True,
        sideforce=True
    )
    assert dyn[-1].x_e == pytest.approx(277.3463197415092, abs=1e-1)
    assert dyn[-1].y_e == pytest.approx(10.791223276048788, abs=1)
    assert dyn[-1].z_e == pytest.approx(3.1722726974414805, abs=1e-1)


def test_final_position_SpeedGammaTrack_with_drag():
    dyn = get_trajectory(
        parameterization=asb.DynamicsPointMass3DSpeedGammaTrack,
        drag=True,
        sideforce=False
    )
    assert dyn[-1].x_e == pytest.approx(277.5774436945314, abs=1e-2)
    assert dyn[-1].z_e == pytest.approx(3.1722727033631886, abs=1e-2)


def test_final_position_SpeedGammaTrack_no_drag():
    dyn = get_trajectory(
        parameterization=asb.DynamicsPointMass3DSpeedGammaTrack,
        drag=False,
        sideforce=False
    )
    assert dyn[-1].x_e == pytest.approx(1000, abs=1e-2)
    assert dyn[-1].z_e == pytest.approx(-509.5, abs=1e-2)


def test_cross_compare_with_drag():
    dyn1 = get_trajectory(
        parameterization=asb.DynamicsPointMass3DCartesian,
        drag=True
    )
    dyn2 = get_trajectory(
        parameterization=asb.DynamicsPointMass3DSpeedGammaTrack,
        drag=True
    )
    assert dyn1[-1].x_e == pytest.approx(dyn2[-1].x_e, abs=1e-2, rel=1e-2)
    assert dyn1[-1].y_e == pytest.approx(dyn2[-1].y_e, abs=1e-2, rel=1e-2)
    assert dyn1[-1].z_e == pytest.approx(dyn2[-1].z_e, abs=1e-2, rel=1e-2)
    assert dyn1[-1].u_e == pytest.approx(dyn2[-1].u_e, abs=1e-2, rel=1e-2)
    assert dyn1[-1].v_e == pytest.approx(dyn2[-1].v_e, abs=1e-2, rel=1e-2)
    assert dyn1[-1].w_e == pytest.approx(dyn2[-1].w_e, abs=1e-2, rel=1e-2)


def test_cross_compare_no_drag():
    dyn1 = get_trajectory(
        parameterization=asb.DynamicsPointMass3DCartesian,
        drag=False
    )
    dyn2 = get_trajectory(
        parameterization=asb.DynamicsPointMass3DSpeedGammaTrack,
        drag=False
    )
    assert dyn1[-1].x_e == pytest.approx(dyn2[-1].x_e, abs=1e-2, rel=1e-2)
    assert dyn1[-1].y_e == pytest.approx(dyn2[-1].y_e, abs=1e-2, rel=1e-2)
    assert dyn1[-1].z_e == pytest.approx(dyn2[-1].z_e, abs=1e-2, rel=1e-2)
    assert dyn1[-1].u_e == pytest.approx(dyn2[-1].u_e, abs=1e-2, rel=1e-2)
    assert dyn1[-1].v_e == pytest.approx(dyn2[-1].v_e, abs=1e-2, rel=1e-2)
    assert dyn1[-1].w_e == pytest.approx(dyn2[-1].w_e, abs=1e-2, rel=1e-2)


#
if __name__ == '__main__':
    dyn = get_trajectory(
        asb.DynamicsPointMass3DSpeedGammaTrack,
        plot=True
    )
    pytest.main()
