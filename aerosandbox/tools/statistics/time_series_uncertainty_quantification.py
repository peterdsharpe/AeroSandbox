import warnings
from typing import Union, Iterable, List, Tuple

from tqdm import tqdm

import aerosandbox.numpy as np
from aerosandbox.tools.pretty_plots.utilities.natural_univariate_spline import NaturalUnivariateSpline as Spline
from scipy import signal
from aerosandbox.tools.code_benchmarking import Timer


def estimate_noise_standard_deviation(
        data: np.ndarray,
        estimator_order: int = None,
) -> float:
    """
    Estimates the standard deviation of the random noise in a time-series dataset.

    Relies on several assumptions:

    - The noise is normally-distributed and independent between samples (i.e. white noise).

    - The noise is stationary and homoscedastic (i.e., the noise standard deviation is constant).

    - The noise is uncorrelated with the signal.

    - The sample rate of the data is significantly higher than the highest-frequency component of the signal. (In
        practice, this ratio need not be more than ~5:1, if higher-order estimators are used. At a minimum, however,
        this ratio must be greater than 2:1, corresponding to the Nyquist frequency.)

    The algorithm used in this function is a highly-optimized version of the math described in this repository,
    part of an upcoming paper: https://github.com/peterdsharpe/aircraft-polar-reconstruction-from-flight-test

    Args:

        data: A 1D NumPy array of time-series data.

        estimator_order: The order of the estimator to use. Higher orders are generally more accurate, up to the
            point where sample error starts to dominate. If None, a reasonable estimator order will be chosen automatically.

    Returns: An estimate of the standard deviation of the data's noise component.

    """
    if len(data) < 2:
        raise ValueError("Data must have at least 2 points.")

    if estimator_order is None:
        estimator_order = int(np.clip(len(data) ** 0.5, 1, 1000))

    ##### Noise Variance Reconstruction #####
    from scipy.special import gammaln
    ln_factorial = lambda x: gammaln(x + 1)

    ### For speed, pre-compute the log-factorial of integers from 1 to estimator_order
    # ln_f = ln_factorial(np.arange(estimator_order + 1))
    ln_f = np.cumsum(
        np.log(
            np.concatenate([
                [1],
                np.arange(1, estimator_order + 1)
            ])
        )
    )

    ### Create a convolutional kernel to vectorize the summation
    log_coeffs = (
            2 * ln_f[estimator_order] - ln_f - ln_f[::-1] - 0.5 * ln_factorial(2 * estimator_order)
    )
    indices = np.nonzero(log_coeffs >= np.log(1e-20) + log_coeffs[estimator_order // 2])[0]
    coefficients = np.exp(log_coeffs[indices[0]:indices[-1] + 1])
    coefficients[::2] *= -1  # Flip the sign on every other coefficient
    coefficients -= np.mean(coefficients)  # Remove any bias introduced by floating-point error

    # sample_stdev = signal.convolve(data, coefficients[::-1], 'valid')
    sample_stdev = signal.oaconvolve(data, coefficients[::-1], 'valid')
    return np.mean(sample_stdev ** 2) ** 0.5


def bootstrap_fits(
        x: np.ndarray,
        y: np.ndarray,
        x_noise_stdev: Union[None, float] = 0.,
        y_noise_stdev: Union[None, float] = None,
        n_bootstraps: int = 2000,
        fit_points: Union[int, Iterable[float], None] = 300,
        spline_degree: int = 3,
        normalize: bool = None,
) -> Union[Tuple[np.ndarray, np.ndarray], List[Spline]]:
    """
    Bootstraps a time-series dataset and fits splines to each bootstrap resample.

    Args:

        x: The independent variable (e.g., time) of the dataset. A 1D NumPy array.

        y: The dependent variable (e.g., altitude) of the dataset. A 1D NumPy array.

        n_bootstraps: The number of bootstrap resamples to create.

        fit_points: An optional variable that determines what to do with the splines after they are fit:

            - If an integer, the splines will be evaluated at a linearly-spaced vector of points between the minimum
                and maximum x-values of the dataset, with the number of points equal to `fit_points`. This is the default.

            - If an iterable of floats (e.g. a 1D NumPy array), the splines will be evaluated at those points.

            - If None, the splines won't be evaluated, and instead the splines are returned directly.

        spline_degree: The degree of the splines to fit.

        normalize: Whether or not to normalize the data before fitting. If True, the data will be normalized to
            the range [0, 1] before fitting, and the splines will be un-normalized before being returned. If False,
            the data will not be normalized before fitting.

            - If None (the default), the data will be normalized if and only if `fit_points` is not None.


    Returns: One of the following, depending on the value of `fit_points`:

        - If `fit_points` is an integer or array, then this function returns a tuple of NumPy arrays:

            - `x_fit`: A 1D NumPy array with the x-values at which the splines were evaluated.

            - `y_bootstrap_fits`: A 2D NumPy array of shape (n_bootstraps, len(x_fit)) with the y-values of the
                splines evaluated at each bootstrap resample and at each x-value.

        - If `fit_points` is None, then this function returns a list of `n_bootstraps` splines, each of which is a
            `NaturalUnivariateSpline`, which is a subclass of `scipy.interpolate.UnivariateSpline` with more sensible
            extrapolation.

    """
    ### Set defaults
    if normalize is None:
        normalize = fit_points is not None

    ### Discard any NaN points
    isnan = np.logical_or(
        np.isnan(x),
        np.isnan(y),
    )
    x = x[~isnan]
    y = y[~isnan]

    ### Compute the standard deviation of the noise
    if x_noise_stdev is None:
        x_noise_stdev = estimate_noise_standard_deviation(x)
        print(f"Estimated x-component of noise standard deviation: {x_noise_stdev}")
    if y_noise_stdev is None:
        y_noise_stdev = estimate_noise_standard_deviation(y)
        print(f"Estimated y-component of noise standard deviation: {y_noise_stdev}")

    ### Sort the data by x-value
    sort_indices = np.argsort(x)
    x = x[sort_indices]
    y = y[sort_indices]

    ### Prepare for normalization
    x_min = np.min(x)
    x_max = np.max(x)
    x_rng = x_max - x_min

    y_min = np.min(y)
    y_max = np.max(y)
    y_rng = y_max - y_min

    if normalize:
        x_normalize = lambda x: (x - x_min) / x_rng
        y_normalize = lambda y: (y - y_min) / y_rng
        # x_unnormalize = lambda x_n: x_n * x_rng + x_min
        y_unnormalize = lambda y_n: y_n * y_rng + y_min

        x_stdev_normalized = x_noise_stdev / x_rng
        y_stdev_normalized = y_noise_stdev / y_rng

    else:
        x_normalize = lambda x: x
        y_normalize = lambda y: y
        # x_unnormalize = lambda x_n: x_n
        y_unnormalize = lambda y_n: y_n

        x_stdev_normalized = x_noise_stdev
        y_stdev_normalized = y_noise_stdev

    with tqdm(total=n_bootstraps, desc="Bootstrapping", unit=" samples") as progress_bar:
        splines = []
        n_valid_splines = 0
        n_attempted_splines = 0

        while n_valid_splines < n_bootstraps:

            n_attempted_splines += 1

            ### Obtain a bootstrap resample
            indices = np.random.choice(len(x), size=len(x), replace=True)

            x_sample = x[indices] + np.random.normal(scale=x_noise_stdev, size=len(x))
            y_sample = y[indices]

            order = np.argsort(x_sample)
            x_sample = x_sample[order]
            y_sample = y_sample[order]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                spline = Spline(
                    x=x_normalize(x_sample),
                    y=y_normalize(y_sample),
                    w=np.ones_like(x) / y_stdev_normalized,
                    s=len(x),
                    k=spline_degree,
                )

            if not np.isnan(spline(x_normalize((x_min + x_max) / 2))):
                n_valid_splines += 1
                progress_bar.update(1)
                splines.append(spline)
            else:
                continue

    if fit_points is None:
        return splines

    else:
        ### Determine which x-points to resample at
        if fit_points is None:
            x_fit = None
            if normalize:
                raise ValueError("If `fit_points` is None, `normalize` must be False.")
        elif isinstance(fit_points, int):
            x_fit = np.linspace(
                np.min(x),
                np.max(x),
                fit_points
            )
        else:
            x_fit = np.array(fit_points)

        ### Evaluate the splines at the x-points
        y_bootstrap_fits = np.array([
            y_unnormalize(spline(x_normalize(x_fit)))
            for spline in splines
        ])

        ### Throw an error if all of the splines are NaN
        if np.all(np.isnan(y_bootstrap_fits)):
            raise ValueError("All of the splines are NaN. This is likely due to a poor choice of `spline_degree`.")

        return x_fit, y_bootstrap_fits


if __name__ == '__main__':
    np.random.seed(1)
    N = 1000
    f_sample_over_f_signal = 1000

    t = np.arange(N)
    y = np.sin(2 * np.pi / f_sample_over_f_signal * t) + 0.1 * np.random.randn(len(t))

    print(estimate_noise_standard_deviation(y, 1))

    # d = dict(np.load("raw_data.npz"))
    #
    # x = d["airspeed"]
    # y = d["voltage"] * d["current"]
    #
    # # estimate_noise_standard_deviation(x)
    # #
    # # x_fit, y_bootstrap_fits = bootstrap_fits(
    # #     x, y,
    # #     x_stdev=None,
    # #     y_stdev=None,
    # #     n_bootstraps=20,
    # #     spline_degree=5,
    # # )
    # import matplotlib.pyplot as plt
    # import aerosandbox.tools.pretty_plots as p
    #
    # fig, ax = plt.subplots(figsize=(7, 4))
    #
    # p.plot_with_bootstrapped_uncertainty(
    #     x, y,
    #     x_stdev=None,
    #     y_stdev=estimate_noise_standard_deviation(y[np.argsort(x)]),
    #     ci=[0.75, 0.95],
    #     color="coral",
    #     n_bootstraps=100,
    #     n_fit_points=200,
    #     # ci_to_alpha_mapping=lambda ci: 0.4,
    #     normalize=False,
    #     spline_degree=3,
    # )
    # plt.xlim(x.min(), x.max())
    # plt.ylim(-10, 800)
    # p.set_ticks(1, 0.25, 100, 25)
    # plt.legend(
    #     loc="lower right"
    # )
    # p.show_plot(
    #     xlabel="Cruise Airspeed [m/s]",
    #     ylabel="Electrical Power Required [W]",
    #     title="Raw Data",
    #     legend=False,
    #     dpi=300
    # )
