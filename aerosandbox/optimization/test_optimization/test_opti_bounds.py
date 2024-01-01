import aerosandbox as asb
import pytest


def test_bounds():
    opti = asb.Opti()

    x = opti.variable(init_guess=5, lower_bound=3)
    opti.minimize(x ** 2)

    sol = opti.solve()

    assert sol(x) == pytest.approx(3)


if __name__ == '__main__':
    pytest.main()
