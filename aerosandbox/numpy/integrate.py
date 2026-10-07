"""Integration functions for the AeroSandbox NumPy-like interface.

This module provides quadrature and ODE-integration functions that work with
both NumPy inputs (via scipy.integrate) and CasADi symbolic expressions (via
CasADi's CVODES integrator). Attributes not defined here fall back to
scipy.integrate.
"""

import aerosandbox.numpy as np
import casadi as _cas
import numpy as _onp
from typing import Callable, Literal, Sequence
from scipy import integrate as _scipy_integrate


def __getattr__(name: str):
    """Look up any attribute not defined by this module on scipy.integrate.

    Historically, `aerosandbox.numpy.integrate` leaked the name `integrate`
    (bound to scipy.integrate) into the `aerosandbox.numpy` namespace via the
    star-import in aerosandbox/numpy/__init__.py, shadowing this submodule.
    This fallback keeps access patterns like `np.integrate.trapezoid` or
    `np.integrate.odeint` working, while `np.integrate.quad` and
    `np.integrate.solve_ivp` now resolve to the dual-backend (NumPy + CasADi)
    implementations defined here.
    """
    try:
        return getattr(_scipy_integrate, name)
    except AttributeError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r} "
            f"(also tried scipy.integrate)"
        ) from None


def quad(
    func: Callable | _cas.MX,
    a: float,
    b: float,
    full_output: bool = False,
    variable_of_integration: _cas.MX | None = None,
    **kwargs,
) -> tuple[float, float] | tuple[float, float, dict]:
    """Compute a definite integral, analogous to scipy.integrate.quad.

    If ``func`` is a CasADi expression, it is integrated using CasADi's CVODES
    integrator (allowing symbolic differentiation through the result); otherwise,
    the call is passed through to scipy.integrate.quad.

    Parameters
    ----------
    func : Callable | casadi.MX
        The integrand: either a Python callable (as in scipy.integrate.quad) or a
        CasADi expression.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.
    full_output : bool, optional
        If True, also return additional integrator output. Default is False.
    variable_of_integration : casadi.MX, optional
        [CasADi backend only] The variable to integrate with respect to. Required
        if ``func`` contains more than one symbolic variable.
    **kwargs
        Additional keyword arguments, passed through to scipy.integrate.quad.
        Only allowed for the non-CasADi case.

    Returns
    -------
    tuple[float, float] | tuple[float, float, dict]
        The integral of ``func`` from ``a`` to ``b``, and an estimate of the
        absolute error. If ``full_output`` is True, additional integrator output
        is returned as a third element.

    See Also
    --------
    scipy.integrate.quad : The underlying NumPy implementation.
    """
    if np.is_casadi_type(func):
        if kwargs:
            raise TypeError(
                f"Got unexpected keyword arguments for a CasADi-type `func`: {list(kwargs.keys())}\n"
                f"(Extra keyword arguments are only passed through to scipy.integrate.quad, "
                f"which handles the non-CasADi case.)"
            )
        all_vars = _cas.symvar(func)  # All variables found in the expression graph

        if variable_of_integration is None:
            if not len(all_vars) == 1:
                raise ValueError(
                    f"`func` must be a function of one variable, or you must specify the `variable_of_integration`.\n"
                    f"Currently, it is a function of: {all_vars}"
                )
            variable_of_integration = all_vars[0]

        parameters = [
            var for var in all_vars if not _cas.is_equal(var, variable_of_integration)
        ]

        integrator = _cas.integrator(
            "integrator",
            "cvodes",
            {
                "x": _cas.MX.sym("dummy_variable"),
                "p": _cas.vertcat(*parameters),
                "t": variable_of_integration,
                "ode": func,
            },
            a,  # t0
            b,  # tf
            {  # Options
                "abstol": 1e-8,
                "reltol": 1e-6,
            },
        )
        res = integrator(
            x0=0,
            p=_cas.vertcat(*parameters),
        )
        tol = 1e-8

        if full_output:
            return res["xf"], tol, res
        else:
            return res["xf"], tol

    else:
        return _scipy_integrate.quad(
            func=func,
            a=a,
            b=b,
            full_output=full_output,
            **kwargs,
        )


def solve_ivp(
    fun: Callable | _cas.MX,
    t_span: tuple[float, float],
    y0: np.ndarray | _cas.MX,
    method: Literal["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"] = "RK45",
    t_eval: np.ndarray | _cas.MX | None = None,
    dense_output: bool = False,
    events: Callable | Sequence[Callable] | None = None,
    vectorized: bool = False,
    args: tuple | None = None,
    t_variable: _cas.MX | None = None,
    y_variables: _cas.MX | tuple[_cas.MX] | None = None,
    **options,
):
    """Solve an initial value problem for a system of ODEs.

    This function wraps scipy.integrate.solve_ivp for NumPy functions and provides
    a CasADi-compatible implementation for symbolic differentiation through ODEs.

    Analogous to scipy.integrate.solve_ivp, with additional support for CasADi types.

    Parameters
    ----------
    fun : Callable | casadi.MX
        Right-hand side of the system. The calling signature depends on the backend:

        - For NumPy functions: ``fun(t, y, *args)``, where ``t`` is a scalar and
          ``y`` is an ndarray with shape (n,). Must return an array_like with
          shape (n,).
        - For CasADi symbolic: either a Callable that returns CasADi expressions,
          or a CasADi expression directly. If providing an expression, you must
          also provide ``t_variable``.
    t_span : tuple[float, float]
        Interval of integration (t0, tf). The solver starts at t0 and integrates
        until it reaches tf.
    y0 : ndarray | casadi.MX
        Initial state. Array of shape (n,) or CasADi MX.
    method : {'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'}, optional
        Integration method to use. Only applies to the NumPy backend. Options are:

        - 'RK45' (default): Explicit Runge-Kutta method of order 5(4). Good
          general-purpose solver.
        - 'RK23': Explicit Runge-Kutta method of order 3(2). Faster but less
          accurate than RK45.
        - 'DOP853': Explicit Runge-Kutta method of order 8. High accuracy for
          smooth problems.
        - 'Radau': Implicit Runge-Kutta method of order 5. Good for stiff problems.
        - 'BDF': Implicit multi-step variable-order (1 to 5) method. Good for
          stiff problems.
        - 'LSODA': Adams/BDF method with automatic stiffness detection.

        Note: for the CasADi backend (symbolic differentiation), this parameter is
        ignored and the solver uses CVODES internally.
    t_eval : ndarray | casadi.MX, optional
        Times at which to store the computed solution. If None (default), uses
        solver-internal time points. For the CasADi backend, if None, returns the
        solution at 100 evenly-spaced points.
    dense_output : bool, optional
        Whether to compute a continuous solution. Only supported for the NumPy
        backend. Default is False.
    events : Callable | Sequence[Callable], optional
        Event functions to track. Only supported for the NumPy backend.
    vectorized : bool, optional
        Whether ``fun`` may be called in a vectorized fashion. Only applies to the
        NumPy backend. Default is False.
    args : tuple, optional
        Additional arguments to pass to ``fun``. Only supported for the NumPy
        backend.
    t_variable : casadi.MX, optional
        [CasADi backend only] If ``fun`` is a CasADi expression (not a Callable),
        you must specify which variable represents time.
    y_variables : casadi.MX | tuple[casadi.MX], optional
        [CasADi backend only] The state variables. If None, inferred automatically
        as all variables in ``fun`` except ``t_variable``.
    **options
        Additional options to pass to the solver.

    Returns
    -------
    OdeResult
        Object with the following fields:

        - t: array of time points
        - y: array of solution values at each time point
        - sol: (NumPy backend only) interpolating function for the solution
        - t_events, y_events: (NumPy backend only) event information
        - nfev, njev, nlu: (NumPy backend only) solver statistics
        - status: 0 for success
        - message: human-readable status description
        - success: boolean indicating whether the solver succeeded

    See Also
    --------
    scipy.integrate.solve_ivp : The underlying NumPy implementation.

    Examples
    --------
    Exponential decay:

    >>> def exponential_decay(t, y):
    ...     return -0.5 * y
    >>> sol = solve_ivp(exponential_decay, t_span=(0, 10), y0=[2.5])

    Lorenz system:

    >>> def lorenz(t, y):
    ...     sigma, rho, beta = 10, 28, 8/3
    ...     return [sigma * (y[1] - y[0]),
    ...             y[0] * (rho - y[2]) - y[1],
    ...             y[0] * y[1] - beta * y[2]]
    >>> sol = solve_ivp(lorenz, t_span=(0, 40), y0=[0, 1, 1.05])
    """
    # Determine which backend to use
    if np.is_casadi_type(fun, recursive=False):
        backend = "casadi_expr"
    else:
        try:
            # Probe `fun` with an array-converted y0 (scipy also converts y0
            # before calling fun, so e.g. a plain-list y0 must work here) and
            # with `args`, if given.
            f = np.array(fun(t_span[0], np.array(y0), *(args or ())))
            if np.is_casadi_type(f):
                backend = "casadi_func"
            else:
                try:
                    np.asanyarray(f)
                except ValueError:
                    raise ValueError(
                        "If `fun` is not a Callable, it must be a CasADi expression."
                    )
                backend = "numpy_func"
        except TypeError:
            raise TypeError(
                "If `fun` is not a Callable, it must be a CasADi expression."
            )

    # Do some checks
    if backend == "casadi_func" or backend == "numpy_func":
        if t_variable is not None:
            raise ValueError(
                "If `fun` is a Callable, `t_variable` must be None (as it's implied)."
            )
        if y_variables is not None:
            raise ValueError(
                "If `fun` is a Callable, `y_variables` must be None (as they're implied)."
            )

    if backend == "casadi_expr":
        if t_variable is None:
            raise ValueError(
                "If `fun` is a CasADi expression, `t_variable` must be specified (and the y_variables are inferred)."
            )

        all_vars = _cas.symvar(fun)  # All variables found in the expression graph

        # Determine y_variables by selecting all variables that are not t_variable
        if y_variables is None:
            y_variables = np.array(
                [var for var in all_vars if not _cas.is_equal(var, t_variable)]
            )

    if backend == "numpy_func":
        return _scipy_integrate.solve_ivp(
            fun=fun,
            t_span=t_span,
            y0=y0,
            method=method,
            t_eval=t_eval,
            dense_output=dense_output,
            events=events,
            vectorized=vectorized,
            args=args,
            **options,
        )
    elif backend == "casadi_func" or backend == "casadi_expr":
        # Exception on non-implemented options
        if dense_output:
            raise NotImplementedError(
                "dense_output is not yet implemented for CasADi functions."
            )
        if events is not None:
            raise NotImplementedError(
                "Events are not yet implemented for CasADi functions."
            )
        if args:
            raise NotImplementedError(
                "args are not yet implemented for CasADi functions."
            )

        if not np.is_casadi_type(y0, recursive=False):
            y0 = _cas.vertcat(*y0)

        if backend == "casadi_func":
            t_variable = _cas.MX.sym("t")
            y_variables = _cas.MX.sym("y", y0.shape[0], y0.shape[1])
            fun = np.array(fun(t_variable, y_variables))

        """
        At this point:
        * `fun` is a CasADi expression (cas.MX)
        * `t_variable` is a CasADi variable (cas.MX)
        * `y_variables` is a CasADi variable (cas.MX), possibly a vector of variables
        """
        assert y_variables is not None  # Type narrowing for type checker

        t0 = t_span[0]
        tf = t_span[1]

        # sim_time = t0 + (tf - t0) * t_variable
        ode = _cas.substitute(
            fun,
            t_variable,  # from normalized time
            # (t_variable - t0) / (tf - t0), # to real time
            t0 + (tf - t0) * t_variable,  # to real time
        ) * (tf - t0)

        # Find parameters by finding all variables in the expression graph that are not t_variable or y_variables
        all_vars = _cas.symvar(ode)  # All variables found in the expression graph

        def variable_is_t_or_y(var):
            return (
                _cas.is_equal(var, t_variable)
                or _cas.is_equal(var, y_variables)
                or any(
                    [
                        _cas.is_equal(var, y_variables[i])
                        for i in range(np.prod(y_variables.shape))
                    ]
                )
            )

        parameters = _cas.vertcat(
            *[var for var in all_vars if not variable_is_t_or_y(var)]
        )

        if t_eval is None:
            simtime_eval = np.linspace(0, 1, 100)
        else:
            # Map the requested times onto normalized time in [0, 1].
            if np.is_casadi_type(t_eval, recursive=True) or np.is_casadi_type(
                [t0, tf], recursive=True
            ):
                raise NotImplementedError(
                    "For the CasADi backend, `t_eval` is only supported when both "
                    "`t_eval` and `t_span` are numeric (not CasADi types). "
                    "Leave `t_eval` as None to get the solution at 100 evenly-spaced points."
                )
            simtime_eval = (_onp.asarray(t_eval, dtype=float).reshape(-1) - t0) / (
                tf - t0
            )
            if _onp.any(simtime_eval < 0) or _onp.any(simtime_eval > 1):
                raise ValueError("Values in `t_eval` are not within `t_span`.")

        # Define the integrator
        integrator = _cas.integrator(
            "integrator",
            "cvodes",
            # 'idas',
            {
                "x": y_variables,
                "p": parameters,
                "t": t_variable,
                "ode": ode,
                "quad": 1,
            },
            0,
            simtime_eval,
            {  # Options
                "abstol": 1e-8,
                "reltol": 1e-6,
            },
        )
        res = integrator(
            x0=y0,
            p=parameters,
        )

        return _scipy_integrate._ivp.ivp.OdeResult(
            t=t0 + (tf - t0) * res["qf"],
            y=res["xf"],
            t_events=None,
            y_events=None,
            nfev=0,
            njev=0,
            nlu=0,
            status=0,
            message="",
            success=True,
            sol=None,
        )

    else:
        raise ValueError(f"Invalid backend: {backend}")


if __name__ == "__main__":
    # t = cas.MX.sym("t")
    # print(
    #     quad(
    #         func=t ** 2,
    #         a=0,
    #         b=1,
    #     )
    # )

    def lotkavolterra_func(t, z):
        a, b, c, d = 1.5, 1, 3, 1
        z = _cas.MX(z)
        x = z[0]
        y = z[1]
        return [a * x - b * x * y, -c * y + d * x * y]

    t_eval = np.linspace(0, 15, 3000)
    tf = _cas.MX.sym("tf")
    # t_eval = np.linspace(0, tf, 100)

    sol = solve_ivp(
        lotkavolterra_func,
        t_span=(t_eval[0], t_eval[-1]),
        # t_eval=t_eval,
        y0=[10, 5],
    )

    z = sol.y
    import matplotlib.pyplot as plt

    plt.plot(
        _cas.evalf(_cas.substitute(sol.t.T, tf, 15)),
        _cas.evalf(_cas.substitute(sol.y.T, tf, 15)),
    )
    plt.xlabel("t")
    plt.legend(["x", "y"], shadow=True)
    plt.title("Lotka-Volterra System")
    plt.show()

    t = _cas.MX.sym("t")
    m = _cas.MX.sym("m")
    n = _cas.MX.sym("n")
    a, b, c, d = 1.5, 1, 3, 1
    lotkavolterra_expr = np.array(
        [
            a * m - b * m * n,
            -c * n + d * m * n,
        ]
    )

    sol = solve_ivp(
        lotkavolterra_expr,
        t_span=(t_eval[0], t_eval[-1]),
        t_eval=t_eval,
        y0=[10, 5],
        t_variable=t,
        # y_variables=[m, n],
    )

    plt.plot(
        _cas.evalf(_cas.substitute(sol.t.T, tf, 15)),
        _cas.evalf(_cas.substitute(sol.y.T, tf, 15)),
    )
    plt.xlabel("t")
    plt.legend(["x", "y"], shadow=True)
    plt.title("Lotka-Volterra System")
    plt.show()
