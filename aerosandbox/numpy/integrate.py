import aerosandbox.numpy as np
import casadi as _cas
from typing import Union, Callable, Tuple, Optional, Dict, Any, List
from scipy import integrate


def quad(
        func: Union[Callable, _cas.MX],
        a: float,
        b: float,
        full_output: bool = False,
        variable_of_integration: _cas.MX = None,
) -> Union[
    Tuple[float, float],
    Tuple[float, float, dict]
]:
    if np.is_casadi_type(func):

        all_vars = _cas.symvar(func)  # All variables found in the expression graph

        if variable_of_integration is None:
            if not len(all_vars) == 1:
                raise ValueError(
                    f"`func` must be a function of one variable, or you must specify the `variable_of_integration`.\n"
                    f"Currently, it is a function of: {all_vars}")
            variable_of_integration = all_vars[0]

        parameters = [
            var for var in all_vars
            if not _cas.is_equal(var, variable_of_integration)
        ]

        integrator = _cas.integrator(
            'integrator',
            'cvodes',
            {
                'x'  : _cas.MX.sym('dummy_variable'),
                'p'  : _cas.vertcat(*parameters),
                't'  : variable_of_integration,
                'ode': func,
            },
            a,  # t0
            b,  # tf
            {  # Options
                'abstol': 1e-8,
                'reltol': 1e-6,
            },
        )
        res = integrator(
            x0=0,
            p=_cas.vertcat(*parameters),
        )
        tol = 1e-8

        if full_output:
            return res['xf'], tol, res
        else:
            return res['xf'], tol

    else:
        return integrate.quad(
            func=func,
            a=a,
            b=b,
            full_output=full_output,
        )


def solve_ivp(
        fun: Union[Callable, _cas.MX],
        t_span: Tuple[float, float],
        y0: Union[np.ndarray, _cas.MX],
        method: str = 'RK45',
        t_eval: Union[np.ndarray, _cas.MX] = None,
        dense_output: bool = False,
        events: Union[Callable, List[Callable]] = None,
        vectorized: bool = False,
        args: Optional[Tuple] = None,
        t_variable: _cas.MX = None,
        y_variables: Union[_cas.MX, Tuple[_cas.MX]] = None,
        **options
):

    # Determine which backend to use
    if np.is_casadi_type(fun, recursive=False):
        backend = "casadi_expr"
    else:
        try:
            f = np.array(fun(t_span[0], y0))
            if np.is_casadi_type(f):
                backend = "casadi_func"
            else:
                try:
                    np.asanyarray(f)
                except ValueError:
                    raise ValueError("If `fun` is not a Callable, it must be a CasADi expression.")
                backend = "numpy_func"
        except TypeError:
            raise TypeError("If `fun` is not a Callable, it must be a CasADi expression.")

    # Do some checks
    if backend == "casadi_func" or backend == "numpy_func":
        if t_variable is not None:
            raise ValueError("If `fun` is a Callable, `t_variable` must be None (as it's implied).")
        if y_variables is not None:
            raise ValueError("If `fun` is a Callable, `y_variables` must be None (as they're implied).")

    if backend == "casadi_expr":
        if t_variable is None:
            raise ValueError(
                "If `fun` is a CasADi expression, `t_variable` must be specified (and the y_variables are inferred).")

        all_vars = _cas.symvar(fun)  # All variables found in the expression graph

        # Determine y_variables by selecting all variables that are not t_variable
        if y_variables is None:
            y_variables = np.array([
                var for var in all_vars
                if not _cas.is_equal(var, t_variable)
            ])

    if backend == "numpy_func":
        return integrate.solve_ivp(
            fun=fun,
            t_span=t_span,
            y0=y0,
            method=method,
            t_eval=t_eval,
            dense_output=dense_output,
            events=events,
            vectorized=vectorized,
            args=args,
            **options
        )
    elif backend == "casadi_func" or backend == "casadi_expr":

        # Exception on non-implemented options
        if dense_output:
            raise NotImplementedError("dense_output is not yet implemented for CasADi functions.")
        if events is not None:
            raise NotImplementedError("Events are not yet implemented for CasADi functions.")
        if args:
            raise NotImplementedError("args are not yet implemented for CasADi functions.")

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
                    _cas.is_equal(var, t_variable) or
                    _cas.is_equal(var, y_variables) or
                    any([_cas.is_equal(var, y_variables[i]) for i in range(np.prod(y_variables.shape))])
            )

        parameters = _cas.vertcat(*[
            var for var in all_vars
            if not variable_is_t_or_y(var)
        ])

        simtime_eval = np.linspace(0, 1, 100)

        # Define the integrator
        integrator = _cas.integrator(
            'integrator',
            'cvodes',
            # 'idas',
            {
                'x'   : y_variables,
                'p'   : parameters,
                't'   : t_variable,
                'ode' : ode,
                'quad': 1,
            },
            0,
            simtime_eval,
            {  # Options
                'abstol': 1e-8,
                'reltol': 1e-6,
            },
        )
        res = integrator(
            x0=y0,
            p=parameters,
        )

        return integrate._ivp.ivp.OdeResult(
            t=t0 + (tf - t0) * res["qf"],
            y=res['xf'],
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


if __name__ == '__main__':
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
    plt.xlabel('t')
    plt.legend(['x', 'y'], shadow=True)
    plt.title('Lotka-Volterra System')
    plt.show()

    t = _cas.MX.sym("t")
    m = _cas.MX.sym("m")
    n = _cas.MX.sym("n")
    a, b, c, d = 1.5, 1, 3, 1
    lotkavolterra_expr = np.array([
        a * m - b * m * n,
        -c * n + d * m * n,
    ])

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
    plt.xlabel('t')
    plt.legend(['x', 'y'], shadow=True)
    plt.title('Lotka-Volterra System')
    plt.show()
