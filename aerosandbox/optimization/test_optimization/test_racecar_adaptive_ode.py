import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.numpy.integrate import solve_ivp
import pytest


def test_racecar_adaptive_ode(plot=False):

    N = 100  # number of control intervals

    opti = asb.Opti()  # Optimization problem

    t_final = opti.variable(init_guess=1, lower_bound=1e-2)
    u_amp = opti.variable(init_guess=np.ones(10), lower_bound=0, upper_bound=1)

    # u_amp = opti.variable(init_guess=np.ones(20) * 0.5, lower_bound=-0.1, upper_bound=0.1)

    # u_amp = opti.variable(init_guess=np.eye(10)[0],)

    def u(t):
        u = 0
        # # Fourier
        # for i in range(np.length(u_amp)):
        #     u += u_amp[i] * np.cos(np.pi * (i) * t / t_final)

        # Bernstein
        from scipy.special import comb
        tn = t / t_final
        for i in range(np.length(u_amp)):
            u += (
                    u_amp[i]
                    * comb(np.length(u_amp) - 1, i)
                    * tn ** i
                    * (1 - tn) ** (np.length(u_amp) - 1 - i)
            )

        # # # Step
        # index = np.floor(t / t_final * np.length(u_amp))
        # for i in range(np.length(u_amp)):
        #     if i == 0:
        #         weight = np.where(index <= 0, 1, 0)
        #     elif i == np.length(u_amp) - 1:
        #         weight = np.where(index >= i, 1, 0)
        #     else:
        #         weight = np.where(index == i, 1, 0)
        #
        #     u += u_amp[i] * weight

        return u

    def func(t, y):
        return np.array([
            y[1],
            u(t) - y[1],
        ])

    res = solve_ivp(
        fun=func,
        t_span=(0, t_final),
        y0=[0, 0],
    )
    t = res.t
    x = res.y[0, :]
    v = res.y[1, :]

    opti.minimize(t_final)

    opti.subject_to([
        v <= 1 - np.sin(2 * np.pi * x) / 2,
        x[-1] >= 1,
    ])

    def callback(i):
        soli = asb.OptiSol(opti, opti.debug)

        print(f"t_final: {soli(t_final)}")

        if i % 5 == 0:
            import matplotlib.pyplot as plt
            import aerosandbox.tools.pretty_plots as p
            fig, ax = plt.subplots()
            ax.plot(soli(t), soli(v), label="speed")
            ax.plot(soli(t), soli(x), label="position")
            ax.plot(soli(t), soli(1 - np.sin(2 * np.pi * x) / 2), "r--", label="speed limit")
            t_plot = np.linspace(0, soli(t_final), 1000)
            ax.plot(t_plot, soli(u(t_plot)), "k", label="throttle")
            plt.ylim(-0.1, 1.6)
            p.show_plot(f"i = {i}")

    sol = opti.solve(
        # callback=callback,
        # verbose=False,
        # behavior_on_failure="return_last"
    )

    assert sol(t_final) == pytest.approx(1.9683594, abs=1e-3)

    if plot:
        import matplotlib.pyplot as plt
        import aerosandbox.tools.pretty_plots as p

        fig, ax = plt.subplots()
        ax.plot(sol(t), sol(v), label="speed")
        ax.plot(sol(t), sol(x), label="position")
        ax.plot(sol(t), sol(1 - np.sin(2 * np.pi * x) / 2), "r--", label="speed limit")
        t_plot = np.linspace(0, sol(t_final), 1000)
        ax.plot(t_plot, sol(u(t_plot)), "k", label="throttle")
        plt.xlabel("Time [s]")
        plt.ylabel("Position [m] / Speed [m/s] / Throttle [-]")
        p.show_plot(
            rotate_axis_labels=False
        )
