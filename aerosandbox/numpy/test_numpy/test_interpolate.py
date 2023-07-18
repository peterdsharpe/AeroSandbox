import aerosandbox.numpy as np
import pytest
import casadi as cas
import numpy as onp


def test_interp():
    x_np = np.arange(5)
    y_np = x_np + 10

    for x, y in zip(
            [x_np, cas.DM(x_np)],
            [y_np, cas.DM(y_np)]
    ):

        assert np.interp(0, x, y) == pytest.approx(10)
        assert np.interp(4, x, y) == pytest.approx(14)
        assert np.interp(0.5, x, y) == pytest.approx(10.5)
        # Extrapolation has different behaviours between casadi and numpy. TODO investigate why
        # assert np.interp(-1, x, y) == pytest.approx(10)
        # assert np.interp(5, x, y) == pytest.approx(14)
        assert np.interp(-1, x, y, left=-10) == pytest.approx(-10)
        assert np.interp(5, x, y, right=-10) == pytest.approx(-10)
        assert np.interp(5, x, y, period=4) == pytest.approx(11)


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
    value = np.interpn(
        points, values, point
    )
    assert value == pytest.approx(value_func_3d(*point))

    ### CasADi test
    point = cas.DM(point)
    value = np.interpn(
        points, values, point
    )
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

    point = np.array([
        [2.21, 3.12, 1.15],
        [3.42, 0.81, 2.43]
    ])
    value = np.interpn(
        points, values, point
    )
    assert np.all(
        value == pytest.approx(
            value_func_3d(
                *[
                    point[:, i] for i in range(point.shape[1])
                ]
            )
        )
    )
    assert len(value) == 2

    ### CasADi test
    point = cas.DM(point)
    value = np.interpn(
        points, values, point
    )
    value_actual = value_func_3d(
        *[
            np.array(point[:, i]) for i in range(point.shape[1])
        ]
    )
    for i in range(len(value)):
        assert value[i] == pytest.approx(float(value_actual[i]))
    assert value.shape == (2,)


def test_interpn_bspline_casadi():
    """
    The bspline method should interpolate seperable cubic multidimensional polynomials exactly.
    """

    def func(x, y, z):  # Sphere function
        return x ** 3 + y ** 3 + z ** 3

    x = np.linspace(-5, 5, 10)
    y = np.linspace(-5, 5, 20)
    z = np.linspace(-5, 5, 30)
    points = (x, y, z)
    values = func(
        *np.meshgrid(*points, indexing="ij")
    )

    point = np.array([0.4, 0.5, 0.6])
    value = np.interpn(
        points, values, point, method="bspline"
    )

    assert value == pytest.approx(func(*point))


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
        value = np.interpn(
            points, values, point
        )

    ### CasADi test
    point = cas.DM(point)
    with pytest.raises(ValueError):
        value = np.interpn(
            points, values, point
        )


def test_interpn_bounds_error_multiple_samples():
    def value_func_3d(x, y, z):
        return 2 * x + 3 * y - z

    x = np.linspace(0, 5, 10)
    y = np.linspace(0, 5, 20)
    z = np.linspace(0, 5, 30)
    points = (x, y, z)
    values = value_func_3d(*np.meshgrid(*points, indexing="ij"))

    point = np.array([
        [2.21, 3.12, 1.15],
        [3.42, 5.81, 2.43]
    ])
    with pytest.raises(ValueError):
        value = np.interpn(
            points, values, point
        )

    ### CasADi test
    point = cas.DM(point)
    with pytest.raises(ValueError):
        value = np.interpn(
            points, values, point
        )


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
        points, values, point,
        method="bspline",
        bounds_error=False,
        fill_value=-17
    )
    assert value == pytest.approx(-17)

    value = np.interpn(
        points, values, point,
        method="bspline",
        bounds_error=False,
    )
    assert np.isnan(value)

    value = np.interpn(
        points, values, point,
        method="bspline",
        bounds_error=None,
        fill_value=None
    )
    assert value == pytest.approx(value_func_3d(5, 3.12, 1.15))


if __name__ == '__main__':
    pytest.main()
