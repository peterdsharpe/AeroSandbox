import aerosandbox.numpy as np


def estimate_noise_standard_deviation(
        data: np.ndarray,
        reconstructor_order: int = None,
):
    if len(data) < 2:
        raise ValueError("Data must have at least 2 points.")

    if reconstructor_order is None:
        reconstructor_order = min(
            max(
                1,
                len(data) // 2
            ),
            10
        )
    print(reconstructor_order)

    ##### Noise Variance Reconstruction #####
    from scipy.special import comb

    N = len(data)
    d = reconstructor_order

    variance = 0
    for i in range(N - d + 1):
        sample_stdev = 0
        for j in range(d + 1):
            sample_stdev += comb(d, j, exact=True) * (-1) ** j * data[i + j - 1]
        variance += sample_stdev ** 2

    variance /= (N - d) * comb(2 * d, d, exact=True)

    return variance ** 0.5

    # from scipy.special import comb, binom
    # denominator = np.sqrt(comb(2 * reconstructor_order, reconstructor_order))
    #
    # for _ in range(reconstructor_order):
    #     data = np.diff(data) / denominator ** (1 / reconstructor_order)
    #
    # estimated_noise_standard_deviation = np.std(data)
    #
    # return estimated_noise_standard_deviation


import numpy as np
from scipy.special import gammaln


def calculate_sigma(s, d):
    N = len(s)
    f = np.zeros(N - d)
    for i in range(N - d):
        for k in range(d + 1):
            f[i] += ((-1) ** k) * np.exp(gammaln(d + 1) - gammaln(k + 1) - gammaln(d - k + 1)) * s[i + k]
    sigma_squared = np.sum(f ** 2) / (np.exp(gammaln(2 * d + 1) - 2 * gammaln(d + 1)) * (N - d))
    return sigma_squared


if __name__ == '__main__':
    np.random.seed(0)
    t = np.linspace(0, 1, 1000)
    y = np.sin(2 * np.pi * t) + 0.1 * np.random.randn(len(t))

    print(estimate_noise_standard_deviation(y))
    print(calculate_sigma(y, 1000) ** 0.5)
