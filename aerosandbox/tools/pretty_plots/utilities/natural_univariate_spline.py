import numpy as np
from scipy import interpolate


class NaturalUnivariateSpline(interpolate.PPoly):
    """
    A Natural UnivariateSpline.

    Identical to a UnivariateSpline, except that extrapolation outside the data range is constrained to be linear.

    Based on: https://bobby.gramacy.com/surrogates/splines.html

    Which paraphrases [Hastie, Tibshirani & Friedman (2017)](https://hastie.su.domains/ElemStatLearn/), Chapters 5, 7, & 8.

    """

    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 w: np.ndarray = None,
                 k: int = 3,
                 s: float = None,
                 ext=None,
                 bbox=None,
                 check_finite=None
                 ):
        """


        Args:

            x: 1-D array of independent input data. Must be increasing; must be strictly increasing if s is 0.

            y: 1-D array of dependent input data, of the same length as x.

            w: Weights for spline fitting. Must be positive. If w is None, weights are all 1. Default is None.

            k: Degree of the smoothing spline. Must be 1 <= k <= 5. k = 3 is a cubic spline. Default is 3.

            s: Positive smoothing factor used to choose the number of knots.

        Returns:



        """
        if s is None:
            m = len(x)
            s = m - (2 * m) ** 0.5  # Identical default to UnivariateSpline's `s` argument.

        ### Deprecate and warn
        import warnings
        if ext is not None:
            warnings.warn(
                "The `ext` argument is deprecated, as a NaturalUnivariateSpline implies extrapolation.",
                DeprecationWarning
            )
        if bbox is not None:
            warnings.warn(
                "The `bbox` argument is deprecated, as a NaturalUnivariateSpline implies extrapolation.",
                DeprecationWarning
            )
        if check_finite is not None:
            warnings.warn(
                "The `check_finite` argument is deprecated.",
                DeprecationWarning
            )

        ### Compute the t, c, and k parameters for a UnivariateSpline
        tck = interpolate.splrep(
            x=x,
            y=y,
            w=w,
            k=k,
            s=s,
        )

        ### Construct the spline, without natural extrapolation
        spline = interpolate.PPoly.from_spline(
            tck=tck
        )

        ### Add spline knots for natural positive extrapolation
        spline.extend(
            c=np.array(
                [[0]] * (k - 2) + [
                    [spline(spline.x[-1], 1)],
                    [spline(spline.x[-1])]
                ]),
            x=np.array([np.Inf])
        )

        ### Add spline knots for natural negative extrapolation
        spline.extend(
            c=np.array(
                [[0]] * (k - 1) + [
                    [spline(spline.x[0], 1)],
                    [spline(spline.x[0])]
                ]),
            x=np.array([spline.x[0]])
        )

        ### Construct the Natural Univariate Spline
        super().__init__(
            c=spline.c,
            x=spline.x,
            extrapolate=True,
        )


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    x = np.linspace(0, 10, 20)
    y = np.sin(x)

    nus = NaturalUnivariateSpline(
        x,
        y,
        k=3,
    )

    t, c, k = interpolate.splrep(
        x,
        y,
        s=len(x) - (2 * len(x)) ** 0.5,
    )

    us = interpolate.PPoly.from_spline((t, c, k))

    x_plot = np.linspace(-5, 15, 5000)

    fig, ax = plt.subplots()
    plt.plot(x, y, ".k", label="Data")
    plt.plot(x_plot, nus(x_plot), "--", label="Natural Univariate Spline")
    plt.plot(x_plot, us(x_plot), "--", label="Univariate Spline")
    p.set_ticks(1, 1, 1, 1)
    p.show_plot()
