from typing import Union, Iterable, List, Tuple

from tqdm import tqdm

import aerosandbox.numpy as np
from aerosandbox.tools.pretty_plots.utilities.natural_univariate_spline import NaturalUnivariateSpline as Spline


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

    The repository is currently private, but will be public at some point; if you would like access to it,
    please contact Peter Sharpe at pds@mit.edu.

    Args:

        data: A 1D NumPy array of time-series data.

        estimator_order: The order of the estimator to use. Higher orders are generally more accurate, up to the
            point where sample error starts to dominate. If None, a reasonable estimator order will be chosen automatically.

    Returns: An estimate of the standard deviation of the data's noise component.

    """
    if len(data) < 2:
        raise ValueError("Data must have at least 2 points.")

    if estimator_order is None:
        estimator_order = min(
            max(
                1,
                len(data) // 4
            ),
            1000
        )

    ##### Noise Variance Reconstruction #####

    N = len(data)
    d = estimator_order

    def f(x):
        """Returns the natural log of the factorial of x."""
        from scipy.special import gammaln
        return gammaln(x + 1)

    f_d = f(d)
    f_2d = f(2 * d)
    ln_N_minus_d = np.log(N - d)

    coefficients = np.zeros(d + 1)
    for j in range(d + 1):
        coefficients[j] = np.exp(
            2 * f_d - f(j) - f(d - j) - 0.5 * (ln_N_minus_d + f_2d)
        ) * (-1) ** j

    sample_stdev = np.convolve(data, coefficients[::-1], 'valid')
    noise_variance = np.sum(sample_stdev ** 2)
    return noise_variance ** 0.5


def bootstrap_fits(
        x: np.ndarray,
        y: np.ndarray,
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

    ### Sort the data
    order = np.argsort(x)
    x = x[order]
    y = y[order]

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

    ### Compute the standard deviation of the noise
    y_stdev = estimate_noise_standard_deviation(y)

    ### Prepare for normalization
    if normalize:
        x_min = np.min(x)
        x_rng = np.max(x) - x_min
        y_min = np.min(y)
        y_rng = np.max(y) - y_min

        x_normalize = lambda x: (x - x_min) / x_rng
        y_normalize = lambda y: (y - y_min) / y_rng
        # x_unnormalize = lambda x_n: x_n * x_rng + x_min
        y_unnormalize = lambda y_n: y_n * y_rng + y_min

        y_stdev_scaled = y_stdev / y_rng

    else:
        x_normalize = lambda x: x
        y_normalize = lambda y: y
        # x_unnormalize = lambda x_n: x_n
        y_unnormalize = lambda y_n: y_n

        y_stdev_scaled = y_stdev

    splines = []

    for _ in tqdm(range(n_bootstraps), desc="Bootstrapping", unit=" samples"):
        ### Obtain a bootstrap resample
        ### Here, instead of truly resampling, we just pick weights that effectively mimic a resample.
        ### A computationally-efficient way to pick weights is the following clever trick with uniform sampling:
        splits = np.random.rand(len(x) + 1) * len(x)  # "limit" bootstrapping
        splits[0] = 0
        splits[-1] = len(x)

        weights = np.diff(np.sort(splits))

        splines.append(
            Spline(
                x=x_normalize(x),
                y=y_normalize(y),
                w=weights / y_stdev_scaled,
                s=len(x),
                k=spline_degree,
                ext='extrapolate'
            )
        )

    if fit_points is None:
        return splines
    else:
        y_bootstrap_fits = np.array([
            y_unnormalize(spline(x_normalize(x_fit)))
            for spline in splines
        ])

        return x_fit, y_bootstrap_fits


if __name__ == '__main__':
    np.random.seed(0)
    N = 1000
    f_sample_over_f_signal = 1000

    t = np.arange(N)
    y = np.sin(2 * np.pi / f_sample_over_f_signal * t) + 0.1 * np.random.randn(len(t))

    print(estimate_noise_standard_deviation(y))
