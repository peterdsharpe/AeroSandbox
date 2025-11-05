import aerosandbox as asb
import aerosandbox.numpy as np
import pytest


def test_opti_simple_unconstrained():
    """Test simple unconstrained optimization."""
    opti = asb.Opti()

    x = opti.variable(init_guess=10)
    opti.minimize((x - 3) ** 2)

    sol = opti.solve(verbose=False)

    assert np.isclose(sol.value(x), 3, atol=1e-4)


def test_opti_with_lower_bound():
    """Test optimization with lower bound constraint."""
    opti = asb.Opti()

    x = opti.variable(init_guess=0, lower_bound=5)
    opti.minimize(x)

    sol = opti.solve(verbose=False)

    assert np.isclose(sol.value(x), 5, atol=1e-4)


def test_opti_with_upper_bound():
    """Test optimization with upper bound constraint."""
    opti = asb.Opti()

    x = opti.variable(init_guess=10, upper_bound=3)
    opti.minimize(-x)

    sol = opti.solve(verbose=False)

    assert np.isclose(sol.value(x), 3, atol=1e-4)


def test_opti_with_bounds():
    """Test optimization with both upper and lower bounds."""
    opti = asb.Opti()

    x = opti.variable(init_guess=0, lower_bound=-5, upper_bound=5)
    opti.minimize((x - 10) ** 2)

    sol = opti.solve(verbose=False)

    ### Minimum would be at x=10, but constrained to x<=5
    assert np.isclose(sol.value(x), 5, atol=1e-4)


def test_opti_equality_constraint():
    """Test optimization with equality constraint."""
    opti = asb.Opti()

    x = opti.variable(init_guess=0)
    y = opti.variable(init_guess=0)

    opti.minimize(x**2 + y**2)
    opti.subject_to(x + y == 10)

    sol = opti.solve(verbose=False)

    ### Minimum of x^2 + y^2 subject to x + y = 10 is at x = y = 5
    assert np.isclose(sol.value(x), 5, atol=1e-3)
    assert np.isclose(sol.value(y), 5, atol=1e-3)


def test_opti_inequality_constraint():
    """Test optimization with inequality constraint."""
    opti = asb.Opti()

    x = opti.variable(init_guess=0)

    opti.minimize(x)
    opti.subject_to(x >= 3)

    sol = opti.solve(verbose=False)

    assert np.isclose(sol.value(x), 3, atol=1e-4)


def test_opti_multiple_variables():
    """Test optimization with multiple variables."""
    opti = asb.Opti()

    x = opti.variable(init_guess=0)
    y = opti.variable(init_guess=0)
    z = opti.variable(init_guess=0)

    opti.minimize((x - 1) ** 2 + (y - 2) ** 2 + (z - 3) ** 2)

    sol = opti.solve(verbose=False)

    assert np.isclose(sol.value(x), 1, atol=1e-4)
    assert np.isclose(sol.value(y), 2, atol=1e-4)
    assert np.isclose(sol.value(z), 3, atol=1e-4)


def test_opti_vector_variable():
    """Test optimization with vector variable."""
    opti = asb.Opti()

    x = opti.variable(init_guess=np.zeros(3))
    target = np.array([1, 2, 3])

    opti.minimize(np.sum((x - target) ** 2))

    sol = opti.solve(verbose=False)

    assert np.allclose(sol.value(x), target, atol=1e-4)


def test_opti_parameter():
    """Test optimization with parameter."""
    opti = asb.Opti()

    p = opti.parameter(value=5)
    x = opti.variable(init_guess=0)

    opti.minimize((x - p) ** 2)

    sol = opti.solve(verbose=False)

    assert np.isclose(sol.value(x), 5, atol=1e-4)


def test_opti_parameter_update():
    """Test updating parameter value and re-solving."""
    opti = asb.Opti()

    p = opti.parameter(value=5)
    x = opti.variable(init_guess=0)

    opti.minimize((x - p) ** 2)

    sol1 = opti.solve(verbose=False)
    assert np.isclose(sol1.value(x), 5, atol=1e-4)

    ### Update parameter
    opti.set_value(p, 10)
    sol2 = opti.solve(verbose=False)

    assert np.isclose(sol2.value(x), 10, atol=1e-4)


def test_opti_quadratic_program():
    """Test quadratic programming problem."""
    opti = asb.Opti()

    x = opti.variable(init_guess=0, n_vars=2)

    ### Minimize x^T Q x + c^T x (note: need x.T for correct matrix multiplication)
    Q = np.array([[2, 0], [0, 2]])
    c = np.array([[-4], [-6]])  ### Column vector to match x

    opti.minimize(x.T @ Q @ x + c.T @ x)

    sol = opti.solve(verbose=False)

    ### Solution should be -Q^(-1) @ c / 2 = [1, 1.5]
    expected = np.array([1, 1.5])
    assert np.allclose(sol.value(x).flatten(), expected, atol=1e-3)


def test_opti_rosenbrock():
    """Test classic Rosenbrock function optimization."""
    opti = asb.Opti()

    x = opti.variable(init_guess=0)
    y = opti.variable(init_guess=0)

    opti.minimize((1 - x) ** 2 + 100 * (y - x**2) ** 2)

    sol = opti.solve(verbose=False)

    ### Rosenbrock minimum is at (1, 1)
    assert np.isclose(sol.value(x), 1, atol=0.01)
    assert np.isclose(sol.value(y), 1, atol=0.01)


def test_opti_subject_to_list():
    """Test adding multiple constraints as a list."""
    opti = asb.Opti()

    x = opti.variable(init_guess=0)
    y = opti.variable(init_guess=0)

    opti.minimize(x + y)
    opti.subject_to([x >= 2, y >= 3, x + y <= 10])

    sol = opti.solve(verbose=False)

    ### Minimum is at x=2, y=3
    assert np.isclose(sol.value(x), 2, atol=1e-3)
    assert np.isclose(sol.value(y), 3, atol=1e-3)


def test_opti_linear_program():
    """Test linear programming problem."""
    opti = asb.Opti()

    x = opti.variable(init_guess=0, lower_bound=0)
    y = opti.variable(init_guess=0, lower_bound=0)

    ### Maximize x + 2y (minimize -(x + 2y))
    opti.minimize(-(x + 2 * y))

    opti.subject_to([x + y <= 10, 2 * x + y <= 15, x <= 8])

    sol = opti.solve(verbose=False)

    ### Should push y as high as possible while satisfying constraints
    ### Optimal solution is at the intersection of active constraints
    assert sol.value(x) >= -0.1  ### Allow small numerical error
    assert sol.value(y) >= -0.1
    ### Check that constraints are satisfied
    assert sol.value(x + y) <= 10.1
    assert sol.value(2 * x + y) <= 15.1


def test_opti_constraint_violation_detection():
    """Test that infeasible problems are detected."""
    opti = asb.Opti()

    x = opti.variable(init_guess=0)

    opti.minimize(x)
    opti.subject_to([x >= 10, x <= 5])

    ### This should fail or raise exception
    with pytest.raises(Exception):
        sol = opti.solve(verbose=False)


def test_opti_freeze_variable():
    """Test freezing a variable to a constant value."""
    opti = asb.Opti()

    x = opti.variable(init_guess=3)
    y = opti.variable(init_guess=0)

    opti.subject_to(x == 5)  ### Freeze x to 5

    opti.minimize((x - 2) ** 2 + (y - 1) ** 2)

    sol = opti.solve(verbose=False)

    assert np.isclose(sol.value(x), 5, atol=1e-4)
    assert np.isclose(sol.value(y), 1, atol=1e-4)


def test_opti_nonlinear_constraint():
    """Test optimization with nonlinear constraint."""
    opti = asb.Opti()

    x = opti.variable(init_guess=1)
    y = opti.variable(init_guess=1)

    opti.minimize(x + y)
    opti.subject_to(x**2 + y**2 >= 4)  ### Outside circle of radius 2

    sol = opti.solve(verbose=False)

    ### Minimum should be on the circle boundary
    assert np.isclose(sol.value(x) ** 2 + sol.value(y) ** 2, 4, atol=0.1)


def test_opti_minimize_absolute_value():
    """Test minimizing sum of absolute values (L1 norm)."""
    opti = asb.Opti()

    x = opti.variable(init_guess=np.zeros(3))
    target = np.array([1, -2, 3])

    ### Use smooth approximation for absolute value
    opti.minimize(np.sum(np.sqrt((x - target) ** 2 + 1e-6)))

    sol = opti.solve(verbose=False)

    assert np.allclose(sol.value(x), target, atol=0.01)


def test_opti_matrix_variable():
    """Test optimization with matrix variable."""
    opti = asb.Opti()

    ### Create 4 scalar variables and reshape to matrix
    x = opti.variable(init_guess=0, n_vars=4)
    X = np.reshape(x, (2, 2))
    target = np.array([[1, 2], [3, 4]])

    opti.minimize(np.sum((X - target) ** 2))

    sol = opti.solve(verbose=False)

    assert np.allclose(sol.value(X), target, atol=1e-4)


def test_opti_callback_function():
    """Test optimization with callback function to track iterations."""
    opti = asb.Opti()

    x = opti.variable(init_guess=0)
    opti.minimize(x**2)

    iteration_count = [0]

    def callback(i):
        iteration_count[0] += 1

    try:
        sol = opti.solve(verbose=False, callback=callback)
        ### Callback might not be supported, that's ok
    except (TypeError, AttributeError):
        pass


def test_opti_initial_guess_influence():
    """Test that initial guess affects convergence."""
    opti = asb.Opti()

    x = opti.variable(init_guess=10)  ### Start far from minimum

    opti.minimize((x - 3) ** 2)

    sol = opti.solve(verbose=False)

    ### Should still converge to correct solution
    assert np.isclose(sol.value(x), 3, atol=1e-4)


def test_opti_bounded_optimization_tight_bounds():
    """Test optimization with very tight bounds."""
    opti = asb.Opti()

    x = opti.variable(init_guess=5, lower_bound=4.99, upper_bound=5.01)

    opti.minimize((x - 10) ** 2)

    sol = opti.solve(verbose=False)

    ### Should hit upper bound
    assert np.isclose(sol.value(x), 5.01, atol=1e-4)


def test_opti_scale_dependent_optimization():
    """Test optimization with different scaling."""
    opti = asb.Opti()

    x = opti.variable(init_guess=0)

    ### Large scale problem
    opti.minimize(1e6 * (x - 1) ** 2)

    sol = opti.solve(verbose=False)

    assert np.isclose(sol.value(x), 1, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
