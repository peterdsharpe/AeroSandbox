from aerosandbox.tools.code_benchmarking import time_function
import aerosandbox as asb
import aerosandbox.numpy as np
import itertools
import matplotlib.patheffects as path_effects
import pytest

eps = 1e-20  # has to be quite large for consistent cvxopt printouts;


def solve_aerosandbox(N=10):
    import aerosandbox as asb
    import aerosandbox.numpy as np

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

    return sol.stats()['n_call_nlp_f']  # return number of function evaluations


def solve_aerosandbox_forced_GP(N=10):
    import aerosandbox as asb
    import aerosandbox.numpy as np

    L = 6  # m, overall beam length
    EI = 1.1e4  # N*m^2, bending stiffness
    q = 110 * np.ones(N)  # N/m, distributed load

    x = np.linspace(0, L, N)  # m, node locations

    opti = asb.Opti()

    w = opti.variable(init_guess=np.ones(N), log_transform=True)  # m, displacement

    th = opti.derivative_of(  # rad, slope
        w, with_respect_to=x,
        derivative_init_guess=np.ones(N),
    )

    M = opti.derivative_of(  # N*m, moment
        th * EI, with_respect_to=x,
        derivative_init_guess=np.ones(N),
    )

    V = opti.derivative_of(  # N, shear
        M, with_respect_to=x,
        derivative_init_guess=np.ones(N),
    )

    opti.constrain_derivative(
        variable=V, with_respect_to=x,
        derivative=q,
    )

    opti.subject_to([
        w[0] >= eps,
        th[0] >= eps,
        M[-1] >= eps,
        V[-1] <= eps,
    ])

    opti.minimize(w[-1])

    sol = opti.solve(verbose=False)
    assert sol(w[-1]) == pytest.approx(1.62, abs=0.01)

    return sol.stats()['n_call_nlp_f']  # return number of function evaluations


def solve_gpkit_cvxopt(N=10):
    import numpy as np
    from gpkit import parse_variables, Model, ureg
    from gpkit.small_scripts import mag

    class Beam(Model):
        """Discretization of the Euler beam equations for a distributed load.

        Variables
        ---------
        EI    [N*m^2]   Bending stiffness
        dx    [m]       Length of an element
        L   5 [m]       Overall beam length

        Boundary Condition Variables
        ----------------------------
        V_tip     eps [N]     Tip loading
        M_tip     eps [N*m]   Tip moment
        th_base   eps [-]     Base angle
        w_base    eps [m]     Base deflection

        Node Variables of length N
        --------------------------
        q  100*np.ones(N) [N/m]    Distributed load
        V                 [N]      Internal shear
        M                 [N*m]    Internal moment
        th                [-]      Slope
        w                 [m]      Displacement

        Upper Unbounded
        ---------------
        w_tip

        """

        @parse_variables(__doc__, globals())
        def setup(self, N=4):
            # minimize tip displacement (the last w)
            self.cost = self.w_tip = w[-1]
            return {
                "definition of dx"        : L == (N - 1) * dx,
                "boundary_conditions"     : [
                    V[-1] >= V_tip,
                    M[-1] >= M_tip,
                    th[0] >= th_base,
                    w[0] >= w_base
                ],
                # below: trapezoidal integration to form a piecewise-linear
                #        approximation of loading, shear, and so on
                # shear and moment increase from tip to base (left > right)
                "shear integration"       :
                    V[:-1] >= V[1:] + 0.5 * dx * (q[:-1] + q[1:]),
                "moment integration"      :
                    M[:-1] >= M[1:] + 0.5 * dx * (V[:-1] + V[1:]),
                # slope and displacement increase from base to tip (right > left)
                "theta integration"       :
                    th[1:] >= th[:-1] + 0.5 * dx * (M[1:] + M[:-1]) / EI,
                "displacement integration":
                    w[1:] >= w[:-1] + 0.5 * dx * (th[1:] + th[:-1])
            }

    b = Beam(N=N, substitutions={"L": 6, "EI": 1.1e4, "q": 110 * np.ones(N)})
    sol = b.solve('cvxopt', verbosity=0)

    assert sol("w")[-1].to('m').magnitude == pytest.approx(1.62, abs=0.01)

    return np.nan


def solve_gpkit_mosek(N=10):
    import numpy as np
    from gpkit import parse_variables, Model, ureg
    from gpkit.small_scripts import mag

    class Beam(Model):
        """Discretization of the Euler beam equations for a distributed load.

        Variables
        ---------
        EI    [N*m^2]   Bending stiffness
        dx    [m]       Length of an element
        L   5 [m]       Overall beam length

        Boundary Condition Variables
        ----------------------------
        V_tip     eps [N]     Tip loading
        M_tip     eps [N*m]   Tip moment
        th_base   eps [-]     Base angle
        w_base    eps [m]     Base deflection

        Node Variables of length N
        --------------------------
        q  100*np.ones(N) [N/m]    Distributed load
        V                 [N]      Internal shear
        M                 [N*m]    Internal moment
        th                [-]      Slope
        w                 [m]      Displacement

        Upper Unbounded
        ---------------
        w_tip

        """

        @parse_variables(__doc__, globals())
        def setup(self, N=4):
            # minimize tip displacement (the last w)
            self.cost = self.w_tip = w[-1]
            return {
                "definition of dx"        : L == (N - 1) * dx,
                "boundary_conditions"     : [
                    V[-1] >= V_tip,
                    M[-1] >= M_tip,
                    th[0] >= th_base,
                    w[0] >= w_base
                ],
                # below: trapezoidal integration to form a piecewise-linear
                #        approximation of loading, shear, and so on
                # shear and moment increase from tip to base (left > right)
                "shear integration"       :
                    V[:-1] >= V[1:] + 0.5 * dx * (q[:-1] + q[1:]),
                "moment integration"      :
                    M[:-1] >= M[1:] + 0.5 * dx * (V[:-1] + V[1:]),
                # slope and displacement increase from base to tip (right > left)
                "theta integration"       :
                    th[1:] >= th[:-1] + 0.5 * dx * (M[1:] + M[:-1]) / EI,
                "displacement integration":
                    w[1:] >= w[:-1] + 0.5 * dx * (th[1:] + th[:-1])
            }

    b = Beam(N=N, substitutions={"L": 6, "EI": 1.1e4, "q": 110 * np.ones(N)})
    sol = b.solve('mosek_conif', verbosity=0)

    assert sol("w")[-1].to('m').magnitude == pytest.approx(1.62, abs=0.01)

    return np.nan


if __name__ == '__main__':

    solvers = {
        "AeroSandbox"          : solve_aerosandbox,
        "AeroSandbox_forced_gp": solve_aerosandbox_forced_GP,
        "GPkit_cvxopt"         : solve_gpkit_cvxopt,
        "GPkit_mosek"          : solve_gpkit_mosek,
    }

    if False:  # If True, runs the benchmark and appends data to respective *.csv files
        for solver_name, solver in solvers.items():
            print(f"Running {solver_name}...")
            solver(N=5)

            N_ideal = 5.0
            Ns_attempted = []

            while True:
                N_ideal *= 1.1

                N = np.round(N_ideal).astype(int)
                if N in Ns_attempted:
                    continue

                print(f"Trying N={N}...")
                Ns_attempted.append(N)

                try:
                    t, nfev = time_function(
                        lambda: solver(N=N),
                    )
                except ValueError:
                    continue
                except KeyboardInterrupt:
                    break
                print(f"{solver_name}: N={N}, t={t}, nfev={nfev}")
                with open(f"{solver_name.lower()}_times.csv", "a") as f:
                    f.write(f"{N},{t},{nfev}\n")

                if N > 10e3:
                    break
                if t > 120:
                    break
                if nfev > 1e6:
                    break

    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p
    import pandas as pd

    fig, ax = plt.subplots(figsize=(5.5, 4))

    fallback_colors = itertools.cycle(p.sns.husl_palette(
        n_colors=len(solvers) - 1,
        h=0, s=0.25, l=0.6,
    ))

    name_remaps = {
        "GPkit_cvxopt"         : "GPkit\n(cvxopt)",
        "GPkit_mosek"          : "GPkit\n(mosek)",
        "AeroSandbox_forced_gp": "AeroSandbox\n(forced to use\nGP formulation)",
    }

    color_remaps = {
        "AeroSandbox": "royalblue",
    }

    notables = ["AeroSandbox"]

    for i, solver_name in enumerate(solvers.keys()):  # For each solver...

        # Reads the data from file
        df = pd.read_csv(f"{solver_name.lower()}_times.csv", header=None, names=["N", "t", "nfev"])
        aggregate_cols = [col for col in df.columns if col != 'N']
        df = df.groupby('N', as_index=False)[aggregate_cols].mean()
        df = df.sort_values('N')

        # Determines which columns to plot
        x = df["N"].values
        y = df["t"].values

        # Figures out which color to use
        if solver_name in color_remaps:
            color = color_remaps[solver_name]
        else:
            color = next(fallback_colors)

        # Plots the raw data
        line, = plt.plot(
            x, y, ".",
            alpha=0.2,
            color=color
        )


        # Makes a curve fit and plots that
        def model(x, p):
            return (
                    p["c"]
                    + np.exp(p["b1"] * np.log(x) + p["a1"])
                    + np.exp(p["b2"] * np.log(x) + p["a2"])
            )


        fit = asb.FittedModel(
            model=model,
            x_data=x,
            y_data=y,
            parameter_guesses={
                "a1": 1,
                "b1": 1,
                "a2": 1,
                "b2": 3,
                "c" : 0,
            },
            parameter_bounds={
                "b1": [0, 10],
                "b2": [0, 10],
                "c" : [0, np.min(y)],
            },
            residual_norm_type="L1",
            put_residuals_in_logspace=True,
            verbose=False
        )
        x_plot = np.geomspace(x.min(), x.max(), 500)
        p.plot_smooth(
            x_plot, fit(x_plot), "-",
            function_of="x",
            color=color,
            alpha=0.8,
            resample_resolution=10000
        )

        # Writes the label for each plot
        if solver_name in name_remaps:
            label_to_write = name_remaps[solver_name]
        else:
            label_to_write = solver_name

        if solver_name in notables:
            ax.annotate(
                label_to_write,
                xy=(x[-1], fit(x[-1])),
                xytext=(-5, -45),
                textcoords="offset points",
                fontsize=10,
                zorder=5,
                alpha=0.9,
                color=color,
                horizontalalignment='right',
                verticalalignment='top',
                path_effects=[
                    path_effects.withStroke(linewidth=2, foreground=ax.get_facecolor(),
                                            alpha=0.8,
                                            ),
                ],
            )
        else:
            ax.annotate(
                label_to_write,
                xy=(x[-1], fit(x[-1])),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=7,
                zorder=4,
                alpha=0.7,
                color=color,
                horizontalalignment='left',
                verticalalignment='center',
                path_effects=[
                    path_effects.withStroke(linewidth=2, foreground=ax.get_facecolor(),
                                            alpha=0.3,
                                            ),
                ]
            )

    plt.xscale("log")
    plt.yscale("log")

    from aerosandbox.tools.string_formatting import eng_string

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: eng_string(x)))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.4g}"))

    p.show_plot(
        "AeroSandbox vs. Disciplined Methods"
        "\nfor the GP-Compatible Beam Problem",
        "Problem Size\n(# of Beam Discretization Points)",
        "Computational\nCost\n\n(Wall-clock\nruntime,\nin seconds)",
        set_ticks=False,
        legend=False,
        dpi=600,
        savefig=["benchmark_gp_beam.pdf", "benchmark_gp_beam.png"]
    )
