from typing import Union, Iterable, Tuple, Optional, Callable
import matplotlib.pyplot as plt

import numpy as np
from aerosandbox.tools.statistics import time_series_uncertainty_quantification as tsuq


def plot_with_bootstrapped_uncertainty(
        x: np.ndarray,
        y: np.ndarray,
        ci: Optional[Union[float, Iterable[float], np.ndarray]] = 0.95,
        x_stdev: Union[None, float] = 0.,
        y_stdev: Union[None, float] = None,
        color: Optional[Union[str, Tuple[float]]] = None,
        draw_data: bool = True,
        label_line: Union[bool, str] = "Best Estimate",
        label_ci: bool = True,
        label_data: Union[bool, str] = "Raw Data",
        line_alpha: float = 0.9,
        ci_to_alpha_mapping: Callable[[float], float] = lambda ci: 0.8 * (1 - ci) ** 0.4,
        n_bootstraps=2000,
        n_fit_points=500,
        spline_degree=3,
        normalize: bool=True,
        x_log_scale: bool = False,
        y_log_scale: bool = False,
):
    x = np.array(x)
    y = np.array(y)

    ### Log-transform the data if desired
    if x_log_scale:
        x = np.log(x)
    if y_log_scale:
        y = np.log(y)

    ### Make sure `ci` is a NumPy array
    if ci is None:
        ci = []
    else:
        try:
            iter(ci)
        except TypeError:
            ci = [ci]
    ci = np.array(ci)

    ### Make sure `ci` is sorted
    ci = np.sort(ci)

    ### Make sure `ci` is in bounds
    if not (np.all(ci > 0) and np.all(ci < 1)):
        raise ValueError("Confidence interval values in `ci` should all be in the range of (0, 1).")

    ### Do the bootstrap fits
    x_fit, y_bootstrap_fits = tsuq.bootstrap_fits(
        x=x,
        y=y,
        x_noise_stdev=x_stdev,
        y_noise_stdev=y_stdev,
        n_bootstraps=n_bootstraps,
        fit_points=n_fit_points,
        spline_degree=spline_degree,
        normalize=normalize,
    )

    ### Undo the log-transform if desired
    if x_log_scale:
        x = np.exp(x)
        x_fit = np.exp(x_fit)
    if y_log_scale:
        y = np.exp(y)
        y_bootstrap_fits = np.exp(y_bootstrap_fits)

    ### Plot the best-estimator line
    line, = plt.plot(
        x_fit,
        np.nanquantile(y_bootstrap_fits, q=0.5, axis=0),
        color=color,
        label=label_line,
        zorder=2,
        alpha=line_alpha,
    )
    if color is None:
        color = line.get_color()

    if x_log_scale:
        plt.xscale('log')
    if y_log_scale:
        plt.yscale('log')

    ### Plot the confidence intervals
    if len(ci) != 0:

        ### Using the method of equal-tails confidence intervals
        lower_quantiles = np.concatenate([[0.5], (1 - ci) / 2])
        upper_quantiles = np.concatenate([[0.5], 1 - (1 - ci) / 2])

        lower_ci = np.nanquantile(y_bootstrap_fits, q=lower_quantiles, axis=0)
        upper_ci = np.nanquantile(y_bootstrap_fits, q=upper_quantiles, axis=0)

        for i, ci_val in enumerate(ci):
            settings = dict(
                color=color,
                alpha=ci_to_alpha_mapping(ci_val),
                linewidth=0,
                zorder=1.5
            )
            plt.fill_between(
                x_fit,
                lower_ci[i],
                lower_ci[i + 1],
                label=f"{ci_val:.0%} CI" if label_ci else None,
                **settings
            )
            plt.fill_between(
                x_fit,
                upper_ci[i],
                upper_ci[i + 1],
                **settings
            )

    ### Plot the data
    if draw_data:
        plt.plot(
            x,
            y,
            ".k",
            label=label_data,
            alpha=0.25,
            markersize=5,
            markeredgewidth=0,
            zorder=1,
        )
    return x_fit, y_bootstrap_fits


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    np.random.seed(0)

    ### Generate data
    x = np.linspace(0, 20, 1001)
    y_true = np.sin(x - 5)  # np.sin(x)

    y_stdev = 0.5

    y_noisy = y_true + y_stdev * np.random.randn(len(x))

    ### Plot spline regression
    fig, ax = plt.subplots(dpi=300)
    x_fit, y_bootstrap_fits = plot_with_bootstrapped_uncertainty(
        x,
        y_noisy,
        ci=[0.75, 0.95],
        label_line="Best Estimate",
        label_data="Data (True Function + Noise)",
    )
    ax.plot(x, y_true, "k--", label="True Function (Hidden)", alpha=0.8, zorder=1)
    plt.legend(ncols=2)

    p.show_plot(
        "Spline Bootstrapping Test",
        r"$x$",
        r"$y$",
        legend=False
    )

    ### Generate data
    x = np.geomspace(10, 1000, 1000)
    y_true = 3 * x ** 0.5

    y_stdev = 0.1

    y_noisy = y_true * y_stdev * np.random.lognormal(size=len(x))

    fig, ax = plt.subplots()
    x_fit, y_bootstrap_fits = plot_with_bootstrapped_uncertainty(
        x,
        y_noisy,
        ci=[0.75, 0.95],
        label_line="Best Estimate",
        label_data="Data (True Function + Noise)",
        # normalize=False,
        x_log_scale=True,
        y_log_scale=True,
    )
    p.show_plot()
