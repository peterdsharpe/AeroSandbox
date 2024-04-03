from aerosandbox.tools.code_benchmarking import time_function
import aerosandbox as asb
import aerosandbox.numpy as np
from scipy import optimize
import itertools
import matplotlib.patheffects as path_effects


# Problem is unimodal for N=2, N=3, and N>=8. Bimodal for 4<=N<=7. Global min is always a vector of ones.

def get_initial_guess(N):
    rng = np.random.default_rng(0)
    return rng.uniform(-10, 10, N)

def objective(x):
    return np.mean(
        100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2
    )

def solve_aerosandbox(N=10):
    opti = asb.Opti()  # set up an optimization environment

    x = opti.variable(init_guess=get_initial_guess(N))
    opti.minimize(objective(x))

    try:
        sol = opti.solve(verbose=False, max_iter=100000000)  # solve
    except RuntimeError:
        raise ValueError(f"N={N} failed!")

    if not np.allclose(sol(x), 1, atol=1e-4):
        raise ValueError(f"N={N} failed!")

    return sol.stats()['n_call_nlp_f']


def solve_scipy_bfgs(N=10):
    res = optimize.minimize(
        fun=objective,
        x0=get_initial_guess(N),
        method="BFGS",
        tol=1e-8,
        options=dict(
            maxiter=np.inf,
        )
    )

    if not np.allclose(res.x, 1, atol=1e-4):
        raise ValueError(f"N={N} failed!")

    return res.nfev


def solve_scipy_slsqp(N=10):
    res = optimize.minimize(
        fun=objective,
        x0=get_initial_guess(N),
        method="SLSQP",
        tol=1e-8,
        options=dict(
            maxiter=1000000000,
        )
    )

    if not np.allclose(res.x, 1, atol=1e-4):
        raise ValueError(f"N={N} failed!")

    return res.nfev


def solve_scipy_nm(N=10):
    res = optimize.minimize(
        fun=objective,
        x0=get_initial_guess(N),
        method="Nelder-Mead",
        options=dict(
            maxiter=np.inf,
            maxfev=np.inf,
            xatol=1e-8,
            adaptive=True,
        )
    )

    if not np.allclose(res.x, 1, atol=1e-4):
        raise ValueError(f"N={N} failed!")

    return res.nfev


def solve_scipy_genetic(N=10):
    res = optimize.differential_evolution(
        func=objective,
        bounds=[(-10, 10)] * N,
        maxiter=1000000000,
        x0=get_initial_guess(N),
    )

    if not np.allclose(res.x, 1, atol=1e-4):
        raise ValueError(f"N={N} failed!")

    return res.nfev


if __name__ == '__main__':

    solvers = {
        "AeroSandbox": solve_aerosandbox,
        "BFGS" : solve_scipy_bfgs,
        "SLSQP": solve_scipy_slsqp,
        "Nelder-Mead": solve_scipy_nm,
        "Genetic"    : solve_scipy_genetic,
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

    fig, ax = plt.subplots(figsize=(5.2, 4))

    fallback_colors = itertools.cycle(p.sns.husl_palette(
        n_colors=len(solvers) - 1,
        h=0, s=0.25, l=0.6,
    ))

    name_remaps = {
        "Nelder-Mead": "Nelder\nMead",
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
        y = df["nfev"].values

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

            # return p["a"] * x ** p["b"] + p["c"]


        fit = asb.FittedModel(
            model=model,
            x_data=x,
            y_data=y,
            parameter_guesses={
                "a1": 0,
                "b1": 2,
                "a2": 1,
                "b2": 3,
                "c": 0,
            },
            parameter_bounds={
                "a1": [0, np.inf],
                "b1": [0, 10],
                "a2": [0, np.inf],
                "b2": [0, 10],
                "c": [0, np.min(y)],
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
    plt.xlim(left=1, right=1e4)
    plt.ylim(bottom=10)

    from aerosandbox.tools.string_formatting import eng_string

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: eng_string(x)))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: eng_string(x)))

    p.show_plot(
        "AeroSandbox vs.\nBlack-Box Optimization Methods",
        # "\nfor the N-Dimensional Rosenbrock Problem",
        "\nProblem Size\n(# of Design Variables)",
        "Computational\nCost\n\n(# of Function\nEvaluations)",
        set_ticks=False,
        legend=False,
        dpi=600,
        savefig=["benchmark_nd_rosenbrock.pdf", "benchmark_nd_rosenbrock.png"]
    )
