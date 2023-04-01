from aerosandbox.tools.pretty_plots.utilities.natural_univariate_spline import NaturalUnivariateSpline as Spline
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def plot_with_bootstrapped_uncertainty(
        x: np.ndarray,
        y: np.ndarray,
        y_stdev: float,
        ci: float = 0.95,
        color=None,
        draw_line=True,
        draw_ci=True,
        draw_data=True,
        label_line=None,
        label_ci=None,
        label_data=None,
        n_bootstraps=2000,
        n_fit_points=300,
        spline_degree=3,
        normalize=True,
):

    if not (ci > 0 and ci < 1):
        raise ValueError("Confidence interval `ci` should be in the range of (0, 1).")

    ### Discard any NaN points
    isnan = np.logical_or(
        np.isnan(x),
        np.isnan(y),
    )
    x = x[~isnan]
    y = y[~isnan]

    ### Prepare for normalization
    if normalize:
        x_min = np.min(x)
        x_rng = np.max(x) - x_min
        y_min = np.min(y)
        y_rng = np.max(y) - y_min

        x_normalize = lambda xi: (xi - x_min) / x_rng
        y_normalize = lambda yi: (yi - y_min) / y_rng
        x_unnormalize = lambda xi: xi * x_rng + x_min
        y_unnormalize = lambda yi: yi * y_rng + y_min

    else:
        x_normalize = lambda xi: xi
        y_normalize = lambda yi: yi
        x_unnormalize = lambda xi: xi
        y_unnormalize = lambda yi: yi

    ### Prepare for the bootstrap
    x_fit = np.linspace(
        np.min(x),
        np.max(x),
        n_fit_points
    )

    y_bootstrap_fits = np.empty((n_bootstraps, len(x_fit)))

    for i in tqdm(range(n_bootstraps), desc="Bootstrapping", unit=" samples"):

        ### Obtain a bootstrap resample
        ### Here, instead of truly resampling, we just pick weights that effectively mimic a resample.
        ### A computationally-efficient way to pick weights is the following clever trick with uniform sampling:
        splits = np.random.rand(len(x) + 1) * len(x)  # "limit" bootstrapping
        splits[0] = 0
        splits[-1] = len(x)

        weights = np.diff(np.sort(splits))

        y_bootstrap_fits[i, :] = y_unnormalize(Spline(
            x=x_normalize(x),
            y=y_normalize(y),
            w=weights / y_normalize(y_stdev),
            s=len(x) * (y_normalize(y).max() - y_normalize(y).min()) ** 0.5,
            k=spline_degree,
            ext='extrapolate'
        )(x_normalize(x_fit)))

    ### Compute a confidence interval using equal-tails method
    y_median_and_ci = np.nanquantile(
        y_bootstrap_fits,
        q=[
            (1 - ci) / 2,
            0.5,
            1 - (1 - ci) / 2
        ],
        axis=0
    )

    if draw_line:
        line, = plt.plot(
            x_fit,
            y_median_and_ci[1, :],
            color=color,
            label=label_line
        )
        if color is None:
            color = line.get_color()

    if draw_ci:
        plt.fill_between(
            x_fit,
            y_median_and_ci[0, :],
            y_median_and_ci[2, :],
            color=color,
            label=label_ci,
            alpha=0.25,
            linewidth=0
        )
    if draw_data:
        line, = plt.plot(
            x,
            y,
            ".",
            color=color,
            label=label_data,
            alpha=0.5
        )
        if color is None:
            color = line.get_color()
    return x_fit, y_bootstrap_fits


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    np.random.seed(0)

    ### Generate data
    x = np.linspace(0, 10, 101)
    y_true = np.abs(x - 5)  # np.sin(x)
    y_noisy = y_true + 0.1 * np.random.randn(len(x))

    ### Plot spline regression
    fig, ax = plt.subplots(dpi=300)
    x_fit, y_bootstrap_fits = plot_with_bootstrapped_uncertainty(
        x,
        y_noisy,
        y_stdev=0.1 * np.std(y_noisy),
        label_line="Best Estimate",
        label_data="Data",
        label_ci="95% CI",
    )
    ax.plot(x, y_true, "k", label="True Function", alpha=0.2)

    p.show_plot(
        "Spline Bootstrapping Test",
        r"$x$",
        r"$y$",
    )
