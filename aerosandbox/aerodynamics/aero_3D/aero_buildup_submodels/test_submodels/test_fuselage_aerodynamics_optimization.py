import aerosandbox as asb
import aerosandbox.numpy as np
import pytest


def test_fuselage_aerodynamics_optimization():
    opti = asb.Opti()

    alpha = opti.variable(init_guess=1, lower_bound=0, upper_bound=30)
    beta = opti.variable(init_guess=1)

    fuselage = asb.Fuselage(
        xsecs=[
            asb.FuselageXSec(
                xyz_c=[xi, 0, 0],
                radius=asb.Airfoil("naca0010").local_thickness(0.8 * xi)
            )
            for xi in np.cosspace(0, 1, 20)
        ],
    )

    from aerosandbox.aerodynamics.aero_3D.aero_buildup_submodels import fuselage_aerodynamics

    aero = fuselage_aerodynamics(
        fuselage,
        op_point=asb.OperatingPoint(
            velocity=10,
            alpha=alpha,
            beta=beta
        )
    )

    opti.minimize(-aero["L"] / aero["D"])
    sol = opti.solve(verbose=False)
    assert sol.value(alpha) == pytest.approx(21.338618, abs=1e-3)
    assert sol.value(beta) == pytest.approx(0, abs=1e-3)

    opti.minimize(aero["D"])
    sol = opti.solve(verbose=False)
    assert sol.value(alpha) == pytest.approx(0, abs=1e-2)
    assert sol.value(beta) == pytest.approx(0, abs=1e-2)


if __name__ == '__main__':
    test_fuselage_aerodynamics_optimization()
    pytest.main()
