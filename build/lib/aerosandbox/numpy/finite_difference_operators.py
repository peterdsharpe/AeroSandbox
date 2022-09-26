from aerosandbox.numpy.array import array, length
import numpy as _onp


def finite_difference_coefficients(
        x: _onp.ndarray,
        x0: float = 0,
        derivative_degree: int = 1,
) -> _onp.ndarray:
    """
    Computes the weights (coefficients) in compact finite differece formulas for any order of derivative
    and to any order of accuracy on one-dimensional grids with arbitrary spacing.

    (Wording above is taken from the paper below, as are docstrings for parameters.)

    Modified from an implementation of:

        Fornberg, Bengt, "Generation of Finite Difference Formulas on Arbitrarily Spaced Grids". Oct. 1988.
        Mathematics of Computation, Volume 51, Number 184, pages 699-706.

        PDF: https://www.ams.org/journals/mcom/1988-51-184/S0025-5718-1988-0935077-0/S0025-5718-1988-0935077-0.pdf

        More detail: https://en.wikipedia.org/wiki/Finite_difference_coefficient

    Args:

        derivative_degree: The degree of the derivative that you are interested in obtaining. (denoted "M" in the
        paper)

        x: The grid points (not necessarily uniform or in order) that you want to obtain weights for. You must
        provide at least as many grid points as the degree of the derivative that you're interested in, plus 1.

            The order of accuracy of your derivative depends in part on the number of grid points that you provide.
            Specifically:

                order_of_accuracy = n_grid_points - derivative_degree

            (This is in general; can be higher in special cases.)

            For example, if you're evaluating a second derivative and you provide three grid points, you'll have a
            first-order-accurate answer.

            (x is denoted "alpha" in the paper)

        x0: The location that you are interested in obtaining a derivative at. This need not be on a grid point.

    Complexity is O(derivative_degree * len(x) ^ 2)

    Returns: A 1D ndarray corresponding to the coefficients that should be placed on each grid point. In other words,
    the approximate derivative at `x0` is the dot product of `coefficients` and the function values at each of the
    grid points `x`.

    """
    ### Check inputs
    if derivative_degree < 1:
        return ValueError("The parameter derivative_degree must be an integer >= 1.")
    expected_order_of_accuracy = length(x) - derivative_degree
    if expected_order_of_accuracy < 1:
        return ValueError("You need to provide at least (derivative_degree+1) grid points in the x vector.")

    ### Implement algorithm; notation from paper in docstring.
    N = length(x) - 1

    delta = _onp.zeros(
        shape=(
            derivative_degree + 1,
            N + 1,
            N + 1
        ),
        dtype="O"
    )

    delta[0, 0, 0] = 1
    c1 = 1
    for n in range(1,
                   N + 1):  # TODO make this algorithm more efficient; we only need to store a fraction of this data.
        c2 = 1
        for v in range(n):
            c3 = x[n] - x[v]
            c2 = c2 * c3
            # if n <= M: # Omitted because d is initialized to zero.
            #     d[n, n - 1, v] = 0
            for m in range(min(n, derivative_degree) + 1):
                delta[m, n, v] = (
                                         (x[n] - x0) * delta[m, n - 1, v] - m * delta[m - 1, n - 1, v]
                                 ) / c3
        for m in range(min(n, derivative_degree) + 1):
            delta[m, n, n] = (
                    c1 / c2 * (
                    m * delta[m - 1, n - 1, n - 1] - (x[n - 1] - x0) * delta[m, n - 1, n - 1]
            )
            )
        c1 = c2

    coefficients_object_array = delta[derivative_degree, -1, :]

    coefficients = array([*coefficients_object_array])  # Reconstructs using aerosandbox.numpy to intelligently type

    return coefficients
