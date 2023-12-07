from aerosandbox.tools.code_benchmarking import time_function
import aerosandbox as asb
import aerosandbox.numpy as np
import itertools
import matplotlib.patheffects as path_effects
import pytest

N = 30

L = 6  # m, overall beam length
EI = 1.1e4  # N*m^2, bending stiffness
q = 110 * np.ones(N)  # N/m, distributed load

x = np.linspace(0, L, N)  # m, node locations

opti = asb.Opti()

w = opti.variable(init_guess=np.zeros(N))  # m, displacement

th = opti.derivative_of(  # rad, slope
    w, with_respect_to=x,
    derivative_init_guess=np.zeros(N),
)

M = opti.derivative_of(  # N*m, moment
    th * EI, with_respect_to=x,
    derivative_init_guess=np.zeros(N),
)

V = opti.derivative_of(  # N, shear
    M, with_respect_to=x,
    derivative_init_guess=np.zeros(N),
)

opti.constrain_derivative(
    variable=V, with_respect_to=x,
    derivative=q,
)

opti.subject_to([
    w[0] == 0,
    th[0] == 0,
    M[-1] == 0,
    V[-1] == 0,
])

sol = opti.solve(verbose=False)

print(sol(w[-1]))
assert sol(w[-1]) == pytest.approx(1.62, abs=0.01)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    w = sol(w)

    fig, ax = plt.subplots(figsize=(3, 1.6))
    plt.plot(
        x,
        w,
        ".-",
        linewidth=2,
        markersize=6,
        zorder=4,
        color="navy"
    )
    from matplotlib import patheffects

    plt.plot(
        [0, 0],
        [0 - 0.5, w[-1]],
        color='gray',
        linewidth=1.5,
        path_effects=[
            patheffects.withTickedStroke()
        ]
    )
    load_scale = 0.5
    for i in range(1, N):
        plt.arrow(
            x[i],
            w[i],
            0,
            q[i] / q.mean() * load_scale,
            width=0.01,
            head_width=0.08,
            color='crimson',
            alpha=0.4,
            length_includes_head=True,
        )

    plt.axis('off')
    p.equal()
    p.show_plot(
        "Cantilever Beam Problem",
        # r"$x$ [m]",
        # r"$w$ [m]",
        savefig="beam_illustration.svg"
    )
