import aerosandbox as asb
import aerosandbox.numpy as np
import pytest

def test_single_point_optimization():

    opti = asb.Opti()

    af = asb.Airfoil("naca0012")

    alpha = opti.variable(init_guess=5, n_vars=1, lower_bound=-30, upper_bound=30)

    aero = af.get_aero_from_neuralfoil(
        alpha=alpha,
        Re=1e6,
        mach=0,
    )

    opti.subject_to(
        aero["CL"] == 0.5
    )

    sol = opti.solve()

    assert sol(alpha) == pytest.approx(4.52, abs=0.5)
    assert sol(aero["CL"]) == pytest.approx(0.5, abs=0.01)

def test_multi_point_optimization():

    opti = asb.Opti()

    af = asb.Airfoil("naca0012")

    alpha = opti.variable(init_guess=5, n_vars=10, lower_bound=-30, upper_bound=30)

    aero = af.get_aero_from_neuralfoil(
        alpha=alpha,
        Re=1e6,
        mach=0,
    )

    opti.subject_to(
        aero["CL"] == 0.5
    )

    sol = opti.solve()


    assert sol(alpha)[0] == pytest.approx(4.52, abs=0.5)
    assert sol(aero["CL"])[0] == pytest.approx(0.5, abs=0.01)

if __name__ == '__main__':
    test_single_point_optimization()
    test_multi_point_optimization()
    pytest.main()