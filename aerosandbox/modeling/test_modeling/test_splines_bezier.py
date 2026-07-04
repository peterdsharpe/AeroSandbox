import aerosandbox.numpy as np
from aerosandbox.modeling.splines.bezier import quadratic_bezier_patch_from_tangents
import casadi as cas
import pytest


def test_quadratic_bezier_patch_basic():
    x, y = quadratic_bezier_patch_from_tangents(
        t=np.linspace(0, 1, 11),
        x_a=1,
        x_b=4,
        y_a=2,
        y_b=3,
        dydx_a=1,
        dydx_b=-30,
    )

    ### Curve should pass through both endpoints
    assert x[0] == pytest.approx(1)
    assert y[0] == pytest.approx(2)
    assert x[-1] == pytest.approx(4)
    assert y[-1] == pytest.approx(3)


def test_quadratic_bezier_patch_parallel_tangents_raises():
    """
    Regression test: parallel end tangents used to divide by zero (ZeroDivisionError for
    float inputs; inf/NaN control points for array inputs). Now raises a clear ValueError.
    """
    with pytest.raises(ValueError, match="parallel"):
        quadratic_bezier_patch_from_tangents(
            t=np.linspace(0, 1, 5),
            x_a=0,
            x_b=10,
            y_a=0,
            y_b=5,
            dydx_a=0.5,
            dydx_b=0.5,  # Parallel to dydx_a
        )


def test_quadratic_bezier_patch_casadi_symbolic_passthrough():
    """Symbolic CasADi inputs cannot be checked numerically, but must not crash the guard."""
    s = cas.MX.sym("s")

    x, y = quadratic_bezier_patch_from_tangents(
        t=0.5,
        x_a=0,
        x_b=10,
        y_a=0,
        y_b=5,
        dydx_a=1 + s,
        dydx_b=-1 - s,
    )

    assert np.is_casadi_type(x, recursive=False)
    assert np.is_casadi_type(y, recursive=False)

    ### Substituting a numeric value in should give the same answer as the all-numeric path
    x_num, y_num = quadratic_bezier_patch_from_tangents(
        t=0.5, x_a=0, x_b=10, y_a=0, y_b=5, dydx_a=1.0, dydx_b=-1.0
    )
    f = cas.Function("f", [s], [x, y])
    x_eval, y_eval = f(0)
    assert float(x_eval) == pytest.approx(x_num)
    assert float(y_eval) == pytest.approx(y_num)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
