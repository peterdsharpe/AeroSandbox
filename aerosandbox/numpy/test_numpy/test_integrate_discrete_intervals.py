import aerosandbox.numpy as np
from aerosandbox.numpy.integrate_discrete import (
    integrate_discrete_intervals,
    integrate_discrete_squared_curvature,
)
import pytest


def test_forward_euler_uniform_spacing():
    """Test forward Euler integration with uniform spacing."""
    x = np.linspace(0, 10, 11)
    f = x**2

    result = integrate_discrete_intervals(f, x, method="forward_euler")
    expected = np.sum(f[:-1] * np.diff(x))

    assert np.allclose(result, f[:-1] * np.diff(x))
    assert np.isclose(np.sum(result), expected)


def test_backward_euler_uniform_spacing():
    """Test backward Euler integration with uniform spacing."""
    x = np.linspace(0, 10, 11)
    f = x**2

    result = integrate_discrete_intervals(f, x, method="backward_euler")
    expected = np.sum(f[1:] * np.diff(x))

    assert np.allclose(result, f[1:] * np.diff(x))
    assert np.isclose(np.sum(result), expected)


def test_trapezoidal_constant_function():
    """Trapezoidal rule should be exact for constant functions."""
    x = np.linspace(0, 5, 50)
    f = np.ones_like(x) * 3.14

    result = integrate_discrete_intervals(f, x, method="trapezoidal")
    integral = np.sum(result)

    expected = 3.14 * (x[-1] - x[0])
    assert np.isclose(integral, expected, rtol=1e-10)


def test_trapezoidal_linear_function():
    """Trapezoidal rule should be exact for linear functions."""
    x = np.linspace(0, 10, 100)
    f = 2 * x + 5

    result = integrate_discrete_intervals(f, x, method="trapezoidal")
    integral = np.sum(result)

    ### Exact integral of (2x + 5) from 0 to 10 is [x^2 + 5x] = 100 + 50 = 150
    expected = 150.0
    assert np.isclose(integral, expected, rtol=1e-10)


def test_simpson_quadratic_function():
    """Simpson's rule should be very accurate for quadratic functions."""
    x = np.linspace(0, 10, 101)
    f = x**2 + 3 * x + 2

    result = integrate_discrete_intervals(f, x, method="forward_simpson")
    integral = np.sum(result)

    ### Exact integral of (x^2 + 3x + 2) from 0 to 10
    ### is [x^3/3 + 3x^2/2 + 2x] = 1000/3 + 150 + 20 = 503.333...
    ### Simpson's rule with endpoint handling introduces small discretization error
    expected = 1000 / 3 + 150 + 20
    assert np.isclose(integral, expected, rtol=1e-4)


def test_backward_simpson_quadratic_function():
    """Backward Simpson's rule should be very accurate for quadratic functions."""
    x = np.linspace(0, 10, 101)
    f = x**2 + 3 * x + 2

    result = integrate_discrete_intervals(f, x, method="backward_simpson")
    integral = np.sum(result)

    ### Simpson's rule with endpoint handling introduces small discretization error
    expected = 1000 / 3 + 150 + 20
    assert np.isclose(integral, expected, rtol=1e-4)


def test_cubic_method_cubic_polynomial():
    """Cubic method should be exact for cubic polynomials."""
    x = np.linspace(0, 5, 50)
    f = x**3 + 2 * x**2 + 3 * x + 1

    result = integrate_discrete_intervals(f, x, method="cubic")
    integral = np.sum(result)

    ### Exact integral: [x^4/4 + 2x^3/3 + 3x^2/2 + x] from 0 to 5
    expected = 5**4 / 4 + 2 * 5**3 / 3 + 3 * 5**2 / 2 + 5
    assert np.isclose(integral, expected, rtol=1e-6)


def test_nonuniform_spacing():
    """Test integration with non-uniform spacing."""
    x = np.array([0, 0.5, 1.5, 3.0, 5.0, 8.0, 10.0])
    f = x**2

    result = integrate_discrete_intervals(f, x, method="trapezoidal")
    integral = np.sum(result)

    ### Compute expected via manual trapezoidal rule
    expected = 0.0
    for i in range(len(x) - 1):
        expected += (f[i] + f[i + 1]) / 2 * (x[i + 1] - x[i])

    assert np.isclose(integral, expected, rtol=1e-10)


def test_multiply_by_dx_false():
    """Test that multiply_by_dx=False returns average function values."""
    x = np.linspace(0, 10, 11)
    f = x**2

    result_with_dx = integrate_discrete_intervals(
        f, x, method="trapezoidal", multiply_by_dx=True
    )
    result_without_dx = integrate_discrete_intervals(
        f, x, method="trapezoidal", multiply_by_dx=False
    )

    dx = np.diff(x)
    assert np.allclose(result_with_dx, result_without_dx * dx)


def test_no_x_specified():
    """Test integration when x is not specified (should use indices)."""
    f = np.array([1, 4, 9, 16, 25])

    result = integrate_discrete_intervals(f, method="trapezoidal")
    expected = integrate_discrete_intervals(
        f, x=np.arange(len(f)), method="trapezoidal"
    )

    assert np.allclose(result, expected)


def test_method_endpoints_lower_order():
    """Test that method_endpoints='lower_order' works correctly."""
    x = np.linspace(0, 10, 50)
    f = np.sin(x)

    result = integrate_discrete_intervals(
        f, x, method="cubic", method_endpoints="lower_order"
    )
    integral = np.sum(result)

    assert len(result) == len(x) - 1
    assert not np.any(np.isnan(result))


def test_method_endpoints_ignore():
    """Test that method_endpoints='ignore' returns fewer intervals for methods needing endpoints."""
    x = np.linspace(0, 10, 50)
    f = np.sin(x)

    ### Test with multiply_by_dx=False
    result_ignore_no_dx = integrate_discrete_intervals(
        f, x, method="forward_simpson", method_endpoints="ignore", multiply_by_dx=False
    )

    ### Test with multiply_by_dx=True (this was the bug - broadcasting error)
    result_ignore_with_dx = integrate_discrete_intervals(
        f, x, method="forward_simpson", method_endpoints="ignore", multiply_by_dx=True
    )

    result_lower = integrate_discrete_intervals(
        f, x, method="forward_simpson", method_endpoints="lower_order"
    )

    ### With 'ignore', forward_simpson gives N-2 intervals (missing last interval due to endpoint)
    ### With 'lower_order', all N-1 intervals are returned
    assert len(result_ignore_no_dx) == len(x) - 2
    assert len(result_ignore_with_dx) == len(x) - 2
    assert len(result_lower) == len(x) - 1

    ### Verify the bug fix: result_ignore_with_dx should be result_ignore_no_dx * dx
    dx = np.diff(x)
    assert np.allclose(result_ignore_with_dx, result_ignore_no_dx * dx[:-1])


def test_sinusoidal_integration_accuracy():
    """Test integration of sinusoidal function with various methods."""
    x = np.linspace(0, 2 * np.pi, 100)
    f = np.sin(x)

    ### Exact integral of sin(x) from 0 to 2*pi is 0
    expected = 0.0

    for method in ["trapezoidal", "forward_simpson", "backward_simpson", "cubic"]:
        result = integrate_discrete_intervals(f, x, method=method)
        integral = np.sum(result)
        assert np.abs(integral - expected) < 0.01, f"Method {method} failed"


def test_exponential_integration():
    """Test integration of exponential function."""
    x = np.linspace(0, 5, 100)
    f = np.exp(x)

    result = integrate_discrete_intervals(f, x, method="trapezoidal")
    integral = np.sum(result)

    ### Exact integral of exp(x) from 0 to 5 is [exp(5) - exp(0)]
    ### Trapezoidal rule has O(h^2) error for smooth functions, so with 100 intervals expect ~0.02% error
    expected = np.exp(5) - 1
    assert np.isclose(integral, expected, rtol=5e-4)


def test_squared_curvature_sine_wave():
    """Test squared curvature integration for sine wave."""
    x = np.linspace(0, 2 * np.pi, 200)
    f = np.sin(x)

    ### Second derivative of sin(x) is -sin(x), so (f'')^2 = sin^2(x)
    ### Integral of sin^2(x) from 0 to 2*pi is pi
    expected = np.pi

    for method in ["cubic", "simpson", "hybrid_simpson_cubic"]:
        result = integrate_discrete_squared_curvature(f, x, method=method)
        integral = np.sum(result)
        assert np.isclose(integral, expected, rtol=0.1), (
            f"Method {method} failed: {integral} vs {expected}"
        )


def test_squared_curvature_parabola():
    """Test squared curvature for a parabola (constant second derivative)."""
    x = np.linspace(0, 10, 100)
    f = x**2

    ### Second derivative of x^2 is 2, so (f'')^2 = 4
    ### Integral of 4 from 0 to 10 is 40
    expected = 40.0

    for method in ["cubic", "simpson", "hybrid_simpson_cubic"]:
        result = integrate_discrete_squared_curvature(f, x, method=method)
        integral = np.sum(result)
        assert np.isclose(integral, expected, rtol=0.05), (
            f"Method {method} failed: {integral} vs {expected}"
        )


def test_squared_curvature_cubic():
    """Test squared curvature for a cubic polynomial."""
    x = np.linspace(0, 5, 100)
    f = x**3

    ### Second derivative of x^3 is 6x, so (f'')^2 = 36x^2
    ### Integral of 36x^2 from 0 to 5 is 36 * [x^3/3]_0^5 = 36 * 125/3 = 1500
    expected = 1500.0

    for method in ["cubic", "simpson", "hybrid_simpson_cubic"]:
        result = integrate_discrete_squared_curvature(f, x, method=method)
        integral = np.sum(result)
        assert np.isclose(integral, expected, rtol=0.05), (
            f"Method {method} failed: {integral} vs {expected}"
        )


def test_invalid_method_raises_error():
    """Test that invalid integration method raises ValueError."""
    x = np.linspace(0, 10, 11)
    f = x**2

    with pytest.raises(ValueError):
        integrate_discrete_intervals(f, x, method="invalid_method")


def test_invalid_method_endpoints_raises_error():
    """Test that invalid method_endpoints raises ValueError."""
    x = np.linspace(0, 10, 50)
    f = x**2

    with pytest.raises(ValueError):
        integrate_discrete_intervals(
            f, x, method="cubic", method_endpoints="invalid_endpoint_method"
        )


def test_periodic_endpoints_not_implemented():
    """Test that periodic endpoints raise NotImplementedError."""
    x = np.linspace(0, 10, 50)
    f = np.sin(x)

    with pytest.raises(NotImplementedError):
        integrate_discrete_intervals(f, x, method="cubic", method_endpoints="periodic")


def test_invalid_squared_curvature_method():
    """Test that invalid squared curvature method raises ValueError."""
    x = np.linspace(0, 10, 50)
    f = x**2

    with pytest.raises(ValueError):
        integrate_discrete_squared_curvature(f, x, method="invalid_method")


def test_midpoint_deprecation_warning():
    """Test that 'midpoint' method raises PendingDeprecationWarning."""
    x = np.linspace(0, 10, 11)
    f = x**2

    with pytest.raises(PendingDeprecationWarning):
        integrate_discrete_intervals(f, x, method="midpoint")


def test_all_method_aliases():
    """Test that all method aliases work correctly."""
    x = np.linspace(0, 5, 50)
    f = x**2

    ### Test forward Euler aliases
    result_forward = integrate_discrete_intervals(f, x, method="forward_euler")
    for alias in ["forward", "euler_forward", "left", "left_riemann"]:
        result_alias = integrate_discrete_intervals(f, x, method=alias)
        assert np.allclose(result_forward, result_alias)

    ### Test backward Euler aliases
    result_backward = integrate_discrete_intervals(f, x, method="backward_euler")
    for alias in ["backward", "euler_backward", "right", "right_riemann"]:
        result_alias = integrate_discrete_intervals(f, x, method=alias)
        assert np.allclose(result_backward, result_alias)

    ### Test trapezoidal aliases
    result_trapz = integrate_discrete_intervals(f, x, method="trapezoidal")
    for alias in ["trapezoid", "trapz"]:
        result_alias = integrate_discrete_intervals(f, x, method=alias)
        assert np.allclose(result_trapz, result_alias)

    ### Test Simpson aliases
    result_simpson = integrate_discrete_intervals(f, x, method="forward_simpson")
    for alias in ["simpson_forward", "simpson"]:
        result_alias = integrate_discrete_intervals(f, x, method=alias)
        assert np.allclose(result_simpson, result_alias)


def test_zero_function():
    """Test integration of zero function returns zeros."""
    x = np.linspace(0, 10, 50)
    f = np.zeros_like(x)

    for method in ["trapezoidal", "forward_simpson", "backward_simpson", "cubic"]:
        result = integrate_discrete_intervals(f, x, method=method)
        assert np.allclose(result, 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
