import aerosandbox as asb
import aerosandbox.numpy as np
import pytest


def test_point_mass_kinetic_energy():
    """
    Point-mass dynamics classes used to raise AttributeError on
    `rotational_kinetic_energy` (and hence `kinetic_energy`), since they
    referenced self.p/q/r, which point masses don't have.
    """
    dyn = asb.DynamicsPointMass1DVertical(
        mass_props=asb.MassProperties(mass=2),
        z_e=-10,
        w_e=3,
    )
    assert dyn.rotational_kinetic_energy == 0
    assert dyn.kinetic_energy == pytest.approx(0.5 * 2 * 3**2)
    assert dyn.translational_kinetic_energy == pytest.approx(0.5 * 2 * 3**2)


def test_point_mass_kinetic_energy_casadi_backend():
    opti = asb.Opti()
    w_e = opti.variable(init_guess=3)
    dyn = asb.DynamicsPointMass1DVertical(
        mass_props=asb.MassProperties(mass=2),
        z_e=-10,
        w_e=w_e,
    )
    KE = dyn.kinetic_energy  # Should not raise
    opti.subject_to(w_e == 3)
    sol = opti.solve(verbose=False)
    assert sol(KE) == pytest.approx(0.5 * 2 * 3**2)


if __name__ == "__main__":
    pytest.main([__file__])
