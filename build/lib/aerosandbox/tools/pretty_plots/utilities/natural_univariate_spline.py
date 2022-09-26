import numpy as np
from scipy import interpolate

class NaturalUnivariateSpline(interpolate.UnivariateSpline):
    """
    A Natural UnivariateSpline.

    Identical to a UnivariateSpline, except that extrapolation outside the data range is constrained to be linear.

    Based on: https://bobby.gramacy.com/surrogates/splines.html

    Which paraphrases [Hastie, Tibshirani & Friedman (2017)](https://hastie.su.domains/ElemStatLearn/), Chapters 5, 7, & 8.

    """

    def __init__(self,
                 x,
                 y,
                 w=None,
                 bbox=[None] * 2,
                 k=3,
                 s=None,
                 ext=0,
                 check_finite=False
                 ):
        super().__init__(
            x=x,
            y=y,
            w=w,
            bbox=bbox,
            k=k,
            s=s,
            ext=ext,
            check_finite=check_finite
        )
        self.x = x
        self.y = y
        self.w = w


    def __call__(self, x, nu=0, ext=None):
        xmin = np.min(self.x)
        xmax = np.max(self.x)

        cubic = super().__call__

        return np.where(
            x < xmin,
            cubic(xmin) + cubic(xmin, nu=1) * (x - xmin),
            np.where(
                x > xmax,
                cubic(xmax) + cubic(xmax, nu=1) * (x - xmax),
                cubic(x)
            )
        )

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    x = np.linspace(0, 10)
    y = np.sin(x)

    interpolator = NaturalUnivariateSpline(
        x,
        y,
        # s=0
    )

    x_plot = np.linspace(-5, 15)
    y_plot = interpolator(x_plot)

    fig, ax = plt.subplots()
    plt.plot(x, y, ".k", label="Data")
    plt.plot(x_plot, y_plot, "--", label="Interpolation")
    p.show_plot()