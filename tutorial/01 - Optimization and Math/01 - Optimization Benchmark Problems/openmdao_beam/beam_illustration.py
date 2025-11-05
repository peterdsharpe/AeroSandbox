import aerosandbox as asb
import aerosandbox.numpy as np

N = 300
E = 1e5
L = 1
b = 0.1
volume = 0.01
tip_load = 1

x = np.linspace(0, L, N)  # m, node locations

opti = asb.Opti()
h = opti.variable(init_guess=np.ones(N), lower_bound=1e-2)
I = b * h**3 / 12

V = np.ones(N) * -tip_load
M = opti.variable(init_guess=np.zeros(N))  # N*m, moment
th = opti.variable(init_guess=np.zeros(N))  # rad, slope
w = opti.variable(init_guess=np.zeros(N))  # m, displacement

opti.subject_to(
    [
        np.diff(M) == np.trapz(V) * np.diff(x),
        np.diff(th) == np.trapz(M / (E * I), modify_endpoints=True) * np.diff(x),
        np.diff(w) == np.trapz(th) * np.diff(x),
    ]
)
opti.subject_to(
    [
        M[-1] == 0,
        th[0] == 0,
        w[0] == 0,
    ]
)
opti.subject_to(np.mean(h * b) <= volume / L)
opti.minimize(tip_load * w[-1])
sol = opti.solve()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    w = sol(w)
    h = sol(h)

    fig, ax = plt.subplots(figsize=(3, 1.6))
    for i in range(N - 1):
        plt.plot(
            [x[i + 1], x[i], x[i + 1]],
            [w[i + 1], w[i], w[i + 1]],
            "-",
            linewidth=(h[i] + h[i + 1]) / 2 * 80,
            zorder=4,
            color="navy",
            solid_joinstyle="bevel",
        )
    from matplotlib import patheffects

    plt.plot(
        [0, 0],
        [0 - 0.1, w[-1]],
        color="gray",
        linewidth=1.5,
        path_effects=[patheffects.withTickedStroke()],
    )
    load_scale = 0.1
    plt.arrow(
        x[-1],
        w[-1],
        0,
        load_scale,
        width=0.01 / 5,
        head_width=0.08 / 3,
        color="crimson",
        alpha=0.8,
        length_includes_head=True,
    )

    plt.axis("off")
    p.equal()
    p.show_plot("Beam Design Problem", dpi=600, savefig="beam_illustration.svg")
