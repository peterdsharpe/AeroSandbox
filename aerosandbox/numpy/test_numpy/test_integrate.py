import aerosandbox.numpy as np
import aerosandbox.numpy.integrate as asb_integrate
import casadi as cas
import pytest
import scipy.integrate


def test_np_integrate_is_the_aerosandbox_submodule():
    """np.integrate should be the dual-backend AeroSandbox submodule
    (regression test: the leaked name `integrate` from `from scipy import
    integrate` used to shadow the submodule, so np.integrate.quad was
    silently scipy's CasADi-incompatible version)."""
    assert np.integrate is asb_integrate
    assert np.integrate.quad is asb_integrate.quad
    assert np.integrate.solve_ivp is asb_integrate.solve_ivp


def test_np_integrate_falls_back_to_scipy_names():
    """Names not defined by the AeroSandbox integrate module should fall back
    to scipy.integrate, preserving historical access patterns."""
    assert np.integrate.trapezoid is scipy.integrate.trapezoid
    assert np.integrate.simpson is scipy.integrate.simpson
    assert np.integrate.odeint is scipy.integrate.odeint
    assert np.integrate.cumulative_trapezoid is scipy.integrate.cumulative_trapezoid

    with pytest.raises(AttributeError):
        np.integrate.this_name_does_not_exist_anywhere


def test_quad_numpy_backend():
    val, err = np.integrate.quad(lambda x: x**2, 0, 1)
    assert val == pytest.approx(1 / 3)

    # scipy-specific keyword arguments should still pass through:
    val, err = np.integrate.quad(
        lambda x, k: k * x**2, 0, 1, args=(2.0,), epsabs=1e-10
    )
    assert val == pytest.approx(2 / 3)


def test_quad_casadi_backend():
    t = cas.MX.sym("t")
    val, err = np.integrate.quad(t**2, 0, 1)
    assert float(val) == pytest.approx(1 / 3, abs=1e-6)

    with pytest.raises(TypeError):
        np.integrate.quad(t**2, 0, 1, epsabs=1e-10)  # scipy-only kwarg


def test_solve_ivp_numpy_backend_scipy_style_call():
    """solve_ivp should accept scipy-style calls: a plain-list y0 and
    `args` (the backend-detection probe used to call fun with the raw y0 and
    without args, raising TypeError)."""

    def exponential_decay(t, y):
        return -0.5 * y

    def parameterized_decay(t, y, k):
        return -k * y

    # (errstate guard: test_all_operations_run.py sets np.seterr(all="raise")
    # at module level, which otherwise leaks into scipy's RK45 internals here.)
    with np.errstate(all="ignore"):
        sol = np.integrate.solve_ivp(exponential_decay, t_span=(0, 10), y0=[2.5])
        assert sol.success
        assert sol.y[0, -1] == pytest.approx(2.5 * np.exp(-0.5 * 10), rel=1e-2)

        sol = np.integrate.solve_ivp(
            parameterized_decay, t_span=(0, 10), y0=[2.5], args=(0.5,)
        )
        assert sol.success
        assert sol.y[0, -1] == pytest.approx(2.5 * np.exp(-0.5 * 10), rel=1e-2)


if __name__ == "__main__":
    pytest.main()
