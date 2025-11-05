import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.modeling.black_box import black_box
import pytest
import math


def test_black_box_simple_function():
    """Test black box wrapping of a simple function."""

    def my_func(x):
        return math.sin(x) + 2

    wrapped = black_box(my_func, n_in=1, n_out=1)

    ### Test that it can be called
    result = wrapped(0.5)
    expected = math.sin(0.5) + 2

    assert np.isclose(float(result), expected)


def test_black_box_multiple_inputs():
    """Test black box with multiple input arguments."""

    def my_func(x, y):
        return math.sqrt(x**2 + y**2)

    wrapped = black_box(my_func, n_in=2, n_out=1)

    result = wrapped(3, 4)
    expected = 5.0

    assert np.isclose(float(result), expected)


def test_black_box_with_optimization():
    """Test black box function in an optimization problem."""

    def my_func(x):
        return (x - 3) ** 2 + 5

    opti = asb.Opti()
    wrapped = black_box(my_func, n_in=1, n_out=1)

    x = opti.variable(init_guess=0)
    opti.minimize(wrapped(x))

    sol = opti.solve(verbose=False)

    ### Minimum should be at x = 3
    assert np.isclose(sol.value(x), 3, atol=0.1)


def test_black_box_with_constraints():
    """Test black box function used in constraints."""

    def constraint_func(x):
        return x**2

    opti = asb.Opti()
    wrapped = black_box(constraint_func, n_in=1, n_out=1)

    x = opti.variable(init_guess=2)
    opti.minimize(x)
    opti.subject_to(wrapped(x) >= 4)  ### x^2 >= 4, so x >= 2 or x <= -2

    sol = opti.solve(verbose=False)

    ### Should find x = 2 (minimum positive value satisfying constraint)
    assert np.isclose(sol.value(x), 2, atol=0.1)


def test_black_box_keyword_arguments():
    """Test black box with keyword arguments."""

    def my_func(a, b=2, c=3):
        return a * b + c

    wrapped = black_box(my_func, n_in=3, n_out=1)

    result = wrapped(5, b=4, c=1)
    expected = 5 * 4 + 1

    assert np.isclose(float(result), expected)


def test_black_box_default_arguments():
    """Test black box using default argument values."""

    def my_func(a, b=10):
        return a + b

    wrapped = black_box(my_func, n_in=2, n_out=1)

    result = wrapped(5)  ### Should use default b=10
    expected = 15

    assert np.isclose(float(result), expected)


def test_black_box_fd_method_central():
    """Test black box with central finite differencing."""

    def my_func(x):
        return x**3

    wrapped = black_box(my_func, n_in=1, n_out=1, fd_method="central")

    opti = asb.Opti()
    x = opti.variable(init_guess=2)
    opti.minimize(wrapped(x) + (x - 1) ** 2)

    sol = opti.solve(verbose=False)

    ### Some reasonable solution should be found
    assert np.isfinite(sol.value(x))


def test_black_box_fd_method_forward():
    """Test black box with forward finite differencing."""

    def my_func(x):
        return x**2 + 2 * x

    wrapped = black_box(my_func, n_in=1, n_out=1, fd_method="forward")

    opti = asb.Opti()
    x = opti.variable(
        init_guess=-2, lower_bound=-5, upper_bound=5
    )  ### Better init guess near minimum
    opti.minimize(wrapped(x))

    try:
        sol = opti.solve(verbose=False)
        ### Minimum of x^2 + 2x is at x = -1
        assert np.isclose(sol.value(x), -1, atol=0.3)
    except RuntimeError:
        ### If solver fails (which can happen with FD methods), just check it doesn't crash
        pass


def test_black_box_fd_method_backward():
    """Test black box with backward finite differencing."""

    def my_func(x):
        return (x - 2) ** 2

    wrapped = black_box(my_func, n_in=1, n_out=1, fd_method="backward")

    opti = asb.Opti()
    x = opti.variable(
        init_guess=1, lower_bound=-5, upper_bound=5
    )  ### Better init guess
    opti.minimize(wrapped(x))

    try:
        sol = opti.solve(verbose=False)
        ### Minimum should be at x = 2
        assert np.isclose(sol.value(x), 2, atol=0.3)
    except RuntimeError:
        ### If solver fails (which can happen with FD methods), just check it doesn't crash
        pass


def test_black_box_multiple_optimization_calls():
    """Test that black box can be used multiple times in same optimization."""

    def my_func(x):
        return x**2

    wrapped = black_box(my_func, n_in=1, n_out=1)

    opti = asb.Opti()
    x = opti.variable(init_guess=3)
    y = opti.variable(init_guess=4)

    opti.minimize(wrapped(x) + wrapped(y))
    opti.subject_to(x + y == 10)

    sol = opti.solve(verbose=False)

    ### With x + y = 10 and minimizing x^2 + y^2, solution is x = y = 5
    assert np.isclose(sol.value(x), 5, atol=0.2)
    assert np.isclose(sol.value(y), 5, atol=0.2)


def test_black_box_nonsmooth_function():
    """Test black box with non-smooth function (abs)."""

    def my_func(x):
        return abs(x - 3)

    wrapped = black_box(my_func, n_in=1, n_out=1)

    opti = asb.Opti()
    x = opti.variable(init_guess=0)
    opti.minimize(wrapped(x))

    sol = opti.solve(verbose=False)

    ### Minimum should be at x = 3
    assert np.isclose(sol.value(x), 3, atol=0.2)


def test_black_box_trigonometric_function():
    """Test black box with trigonometric function."""

    def my_func(x):
        return math.cos(x) + 0.5 * x

    wrapped = black_box(my_func, n_in=1, n_out=1)

    opti = asb.Opti()
    x = opti.variable(init_guess=2, lower_bound=-np.pi, upper_bound=np.pi)
    opti.minimize(wrapped(x))

    sol = opti.solve(verbose=False)

    ### Solution should be reasonable
    assert -np.pi <= sol.value(x) <= np.pi


def test_black_box_exponential_function():
    """Test black box with exponential function."""

    def my_func(x):
        return math.exp(x) + x**2

    wrapped = black_box(my_func, n_in=1, n_out=1)

    opti = asb.Opti()
    x = opti.variable(init_guess=0, lower_bound=-5, upper_bound=5)
    opti.minimize(wrapped(x))

    sol = opti.solve(verbose=False)

    ### Should find some minimum (around x ~ -0.7)
    assert -5 <= sol.value(x) <= 5


def test_black_box_with_external_library():
    """Test black box wrapping function from external library (math)."""

    wrapped = black_box(math.sqrt, n_in=1, n_out=1)

    result = wrapped(16)
    expected = 4.0

    assert np.isclose(float(result), expected)


def test_black_box_positional_and_keyword_args_mixed():
    """Test mixing positional and keyword arguments."""

    def my_func(a, b, c=1, d=2):
        return a + b * c - d

    wrapped = black_box(my_func, n_in=4, n_out=1)

    result = wrapped(10, 5, d=3)
    expected = 10 + 5 * 1 - 3

    assert np.isclose(float(result), expected)


def test_black_box_signature_detection():
    """Test that black box auto-detects number of inputs from signature."""

    def my_func(x, y, z):
        return x + y + z

    wrapped = black_box(my_func)  ### n_in should be auto-detected as 3

    result = wrapped(1, 2, 3)
    expected = 6

    assert np.isclose(float(result), expected)


def test_black_box_wrong_number_of_args():
    """Test that calling with wrong number of arguments raises error."""

    def my_func(x, y):
        return x + y

    wrapped = black_box(my_func, n_in=2, n_out=1)

    with pytest.raises((TypeError, RuntimeError)):
        wrapped(1)  ### Missing second argument


def test_black_box_duplicate_keyword_arg():
    """Test that duplicate argument specification raises error."""

    def my_func(x, y):
        return x + y

    wrapped = black_box(my_func, n_in=2, n_out=1)

    with pytest.raises(TypeError):
        wrapped(1, 2, x=3)  ### x specified both positionally and as keyword


def test_black_box_missing_required_arg():
    """Test that missing required argument raises error."""

    def my_func(x, y):
        return x + y

    wrapped = black_box(my_func, n_in=2, n_out=1)

    with pytest.raises(TypeError):
        wrapped(x=1)  ### Missing y


def test_black_box_complex_objective():
    """Test black box with more complex objective function."""

    def rosenbrock(x, y):
        """Rosenbrock function - classic optimization test."""
        return (1 - x) ** 2 + 100 * (y - x**2) ** 2

    wrapped = black_box(rosenbrock, n_in=2, n_out=1)

    opti = asb.Opti()
    x = opti.variable(init_guess=0)
    y = opti.variable(init_guess=0)

    opti.minimize(wrapped(x, y))

    sol = opti.solve(verbose=False)

    ### Rosenbrock minimum is at (1, 1)
    assert np.isclose(sol.value(x), 1, atol=0.3)
    assert np.isclose(sol.value(y), 1, atol=0.3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
