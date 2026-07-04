import aerosandbox.numpy as np
import pytest
import casadi as cas


def test_interp():
    x_np = np.arange(5)
    y_np = x_np + 10

    for x, y in zip([x_np, cas.DM(x_np)], [y_np, cas.DM(y_np)]):
        assert np.interp(0, x, y) == pytest.approx(10)
        assert np.interp(4, x, y) == pytest.approx(14)
        assert np.interp(0.5, x, y) == pytest.approx(10.5)
        # Extrapolation has different behaviours between casadi and numpy. TODO investigate why
        # assert np.interp(-1, x, y) == pytest.approx(10)
        # assert np.interp(5, x, y) == pytest.approx(14)
        assert np.interp(-1, x, y, left=-10) == pytest.approx(-10)
        assert np.interp(5, x, y, right=-10) == pytest.approx(-10)
        assert np.interp(5, x, y, period=4) == pytest.approx(11)


def test_interp_period_negative_x():
    """interp() with `period` should wrap negative query points into the
    period, matching numpy.interp (regression test: the CasADi branch used
    fmod(), which is negative for negative x, causing extrapolation)."""
    xp = np.linspace(0, 360, 361)
    fp = np.sin(np.radians(xp))

    for x in [-90, -90.0, -450.0, -360.0, 270.0]:
        expected = np.interp(x, xp, fp, period=360)
        result = np.interp(cas.DM(x), xp, fp, period=360)
        assert float(result) == pytest.approx(expected, abs=1e-10), f"{x=}"

    # Same check with the breakpoint data given as CasADi types:
    for x in [-90.0, 270.0]:
        expected = np.interp(x, xp, fp, period=360)
        result = np.interp(x, cas.DM(xp), cas.DM(fp), period=360)
        assert float(result) == pytest.approx(expected, abs=1e-10), f"{x=}"


def test_interpn_linear():
    ### NumPy test

    def value_func_3d(x, y, z):
        return 2 * x + 3 * y - z

    x = np.linspace(0, 5, 10)
    y = np.linspace(0, 5, 20)
    z = np.linspace(0, 5, 30)
    points = (x, y, z)
    values = value_func_3d(*np.meshgrid(*points, indexing="ij"))

    point = np.array([2.21, 3.12, 1.15])
    value = np.interpn(points, values, point)
    assert value == pytest.approx(value_func_3d(*point))

    ### CasADi test
    point = cas.DM(point)
    value = np.interpn(points, values, point)
    assert value == pytest.approx(float(value_func_3d(point[0], point[1], point[2])))


def test_interpn_linear_multiple_samples():
    ### NumPy test

    def value_func_3d(x, y, z):
        return 2 * x + 3 * y - z

    x = np.linspace(0, 5, 10)
    y = np.linspace(0, 5, 20)
    z = np.linspace(0, 5, 30)
    points = (x, y, z)
    values = value_func_3d(*np.meshgrid(*points, indexing="ij"))

    point = np.array([[2.21, 3.12, 1.15], [3.42, 0.81, 2.43]])
    value = np.interpn(points, values, point)
    assert np.all(
        value
        == pytest.approx(value_func_3d(*[point[:, i] for i in range(point.shape[1])]))
    )
    assert np.length(value) == 2

    ### CasADi test
    point = cas.DM(point)
    value = np.interpn(points, values, point)
    value_actual = value_func_3d(
        *[np.array(point[:, i]) for i in range(point.shape[1])]
    )
    for i in range(np.length(value)):
        assert value[i] == pytest.approx(float(value_actual[i]))  # type: ignore[index]
    assert value.shape == (2,)  # type: ignore[union-attr]


def test_interpn_bspline_casadi():
    """
    The bspline method should interpolate seperable cubic multidimensional polynomials exactly.
    """

    def func(x, y, z):  # Sphere function
        return x**3 + y**3 + z**3

    x = np.linspace(-5, 5, 10)
    y = np.linspace(-5, 5, 20)
    z = np.linspace(-5, 5, 30)
    points = (x, y, z)
    values = func(*np.meshgrid(*points, indexing="ij"))

    point = np.array([0.4, 0.5, 0.6])
    value = np.interpn(points, values, point, method="bspline")

    assert value == pytest.approx(func(*point))


def test_interpn_bspline_all_zero_values_shape():
    """The all-zeros-`values` workaround for the CasADi bspline bug should
    return the same shape as a nonzero table would (regression test: it used
    to return zeros with the shape of xi, i.e. (n_points, n_dimensions))."""
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 6)
    points = (x, y)
    values_zero = np.zeros((5, 6))
    values_nonzero = np.ones((5, 6))

    ### Multiple query points
    xi = np.stack([np.linspace(0.1, 0.9, 7), np.linspace(0.2, 0.8, 7)], axis=1)
    result_zero = np.interpn(points, values_zero, xi, method="bspline")
    result_nonzero = np.interpn(points, values_nonzero, xi, method="bspline")
    assert result_zero.shape == result_nonzero.shape == (7,)
    assert result_zero == pytest.approx(0)

    ### Single (scalar-like) query point: both should return a float
    point = np.array([0.5, 0.5])
    result_zero = np.interpn(points, values_zero, point, method="bspline")
    result_nonzero = np.interpn(points, values_nonzero, point, method="bspline")
    assert isinstance(result_zero, float)
    assert isinstance(result_nonzero, float)
    assert result_zero == pytest.approx(0)

    ### CasADi symbolic query points: check that it runs and is zero
    xi_sym = cas.MX.sym("xi", 7, 2)
    result_sym = np.interpn(
        points, values_zero, xi_sym, method="bspline", bounds_error=False
    )
    assert np.length(result_sym) == 7
    assert np.all(cas.DM(np.array(result_sym, dtype=float)) == 0)


def test_interpn_fill_value_None_does_not_mutate_xi():
    """interpn() with fill_value=None used to clamp out-of-bounds points by
    assigning into the caller's xi array in-place (regression test)."""
    x = np.linspace(0, 1, 5)
    points = (x,)
    values = 2 * x

    ### 2D NumPy xi
    xi = np.array([[-0.5], [0.5], [1.5]])
    xi_snapshot = xi.copy()
    result = np.interpn(
        points, values, xi, method="bspline", fill_value=None, bounds_error=False
    )
    assert np.all(xi == xi_snapshot)  # Caller's array must be unchanged
    assert result == pytest.approx([0.0, 1.0, 2.0])  # Clamped extrapolation

    ### 1D NumPy xi (goes through a reshaped view internally)
    xi = np.array([-0.5, 0.5, 1.5])
    xi_snapshot = xi.copy()
    result = np.interpn(
        points, values, xi, method="bspline", fill_value=None, bounds_error=False
    )
    assert np.all(xi == xi_snapshot)
    assert result == pytest.approx([0.0, 1.0, 2.0])

    ### CasADi DM xi
    xi = cas.DM([-0.5, 0.5, 1.5])
    xi_snapshot = cas.DM(xi)
    result = np.interpn(
        points, values, xi, method="bspline", fill_value=None, bounds_error=False
    )
    assert np.all(cas.DM(xi) == xi_snapshot)
    assert result == pytest.approx([0.0, 1.0, 2.0])


def test_interpn_bounds_error_one_sample():
    def value_func_3d(x, y, z):
        return 2 * x + 3 * y - z

    x = np.linspace(0, 5, 10)
    y = np.linspace(0, 5, 20)
    z = np.linspace(0, 5, 30)
    points = (x, y, z)
    values = value_func_3d(*np.meshgrid(*points, indexing="ij"))

    point = np.array([5.21, 3.12, 1.15])
    with pytest.raises(ValueError):
        np.interpn(points, values, point)

    ### CasADi test
    point = cas.DM(point)
    with pytest.raises(ValueError):
        np.interpn(points, values, point)


def test_interpn_bounds_error_multiple_samples():
    def value_func_3d(x, y, z):
        return 2 * x + 3 * y - z

    x = np.linspace(0, 5, 10)
    y = np.linspace(0, 5, 20)
    z = np.linspace(0, 5, 30)
    points = (x, y, z)
    values = value_func_3d(*np.meshgrid(*points, indexing="ij"))

    point = np.array([[2.21, 3.12, 1.15], [3.42, 5.81, 2.43]])
    with pytest.raises(ValueError):
        np.interpn(points, values, point)

    ### CasADi test
    point = cas.DM(point)
    with pytest.raises(ValueError):
        np.interpn(points, values, point)


def test_interpn_fill_value():
    def value_func_3d(x, y, z):
        return 2 * x + 3 * y - z

    x = np.linspace(0, 5, 10)
    y = np.linspace(0, 5, 20)
    z = np.linspace(0, 5, 30)
    points = (x, y, z)
    values = value_func_3d(*np.meshgrid(*points, indexing="ij"))

    point = np.array([5.21, 3.12, 1.15])

    value = np.interpn(
        points, values, point, method="bspline", bounds_error=False, fill_value=-17
    )
    assert value == pytest.approx(-17)

    value = np.interpn(
        points,
        values,
        point,
        method="bspline",
        bounds_error=False,
    )
    assert np.isnan(value)

    value = np.interpn(
        points, values, point, method="bspline", bounds_error=None, fill_value=None
    )
    assert value == pytest.approx(value_func_3d(5, 3.12, 1.15))


if __name__ == "__main__":
    pytest.main()
