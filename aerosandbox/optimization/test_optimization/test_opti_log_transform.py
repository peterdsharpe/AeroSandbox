import aerosandbox as asb
import pytest


def test_bounds():
    opti = asb.Opti()
    x = opti.variable(init_guess=3, log_transform=True, lower_bound=7)
    opti.minimize(x)
    sol = opti.solve()

    assert sol(x) == pytest.approx(7)


def test_negative_lower_bound_raises():
    ### A negative lower bound on a log-transformed variable would silently create
    ### a NaN constraint (and an opaque solver crash); should raise a clear error instead.
    opti = asb.Opti()
    with pytest.raises(ValueError, match="log-transformed"):
        opti.variable(init_guess=1, log_transform=True, lower_bound=-1)


def test_nonpositive_upper_bound_raises():
    ### An upper bound <= 0 on a log-transformed (hence, always-positive) variable can
    ### never be satisfied; should raise a clear error instead of crashing in the solver.
    opti = asb.Opti()
    with pytest.raises(ValueError, match="log-transformed"):
        opti.variable(init_guess=1, log_transform=True, upper_bound=-1)

    with pytest.raises(ValueError, match="log-transformed"):
        opti.variable(init_guess=1, log_transform=True, upper_bound=0)


def test_zero_lower_bound_still_works():
    ### A lower bound of exactly 0 is trivially satisfiable and worked historically;
    ### make sure it stays that way.
    import warnings

    opti = asb.Opti()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # np.log(0) raises a (harmless) RuntimeWarning
        x = opti.variable(init_guess=1, log_transform=True, lower_bound=0)
    opti.minimize((x - 2) ** 2)
    sol = opti.solve(verbose=False)

    assert sol(x) == pytest.approx(2)


if __name__ == "__main__":
    pytest.main()
