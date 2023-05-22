import aerosandbox.numpy as np


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


if __name__ == '__main__':
    np.random.seed(0)
    N = 1000000
    f_sample_over_f_signal = 1000

    t = np.arange(N)
    y = np.sin(2 * np.pi / f_sample_over_f_signal * t) + 0.1 * np.random.randn(len(t))

    print(estimate_noise_standard_deviation(y))
