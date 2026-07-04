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


def test_rigid_body_rotational_kinetic_energy_full_inertia_tensor():
    """
    Rotational kinetic energy must equal 0.5 * w^T @ I @ w, including the
    products of inertia (previously omitted).
    """
    mass_props = asb.MassProperties(
        mass=1,
        Ixx=1.0,
        Iyy=2.0,
        Izz=3.0,
        Ixy=0.1,
        Iyz=0.2,
        Ixz=0.3,
    )
    p, q, r = 0.3, -0.7, 1.1

    dyn = asb.DynamicsRigidBody3DBodyEuler(
        mass_props=mass_props,
        u_b=10,
        p=p,
        q=q,
        r=r,
    )

    w = np.array([p, q, r])
    inertia_tensor = np.array(mass_props.inertia_tensor, dtype=float)
    KE_expected = 0.5 * np.einsum("i,ij,j->", w, inertia_tensor, w)

    assert dyn.rotational_kinetic_energy == pytest.approx(KE_expected)
    assert dyn.kinetic_energy == pytest.approx(
        dyn.translational_kinetic_energy + KE_expected
    )


def test_rigid_body_kinetic_energy_casadi_backend():
    """
    The energy properties should also work symbolically (CasADi backend).
    """
    opti = asb.Opti()
    p = opti.variable(init_guess=0.3)
    q = opti.variable(init_guess=-0.7)
    r = opti.variable(init_guess=1.1)

    mass_props = asb.MassProperties(
        mass=1,
        Ixx=1.0,
        Iyy=2.0,
        Izz=3.0,
        Ixy=0.1,
        Iyz=0.2,
        Ixz=0.3,
    )

    dyn = asb.DynamicsRigidBody3DBodyEuler(
        mass_props=mass_props,
        u_b=10,
        p=p,
        q=q,
        r=r,
    )

    KE = dyn.rotational_kinetic_energy  # Should build a symbolic expression

    opti.subject_to([p == 0.3, q == -0.7, r == 1.1])
    sol = opti.solve(verbose=False)

    w = np.array([0.3, -0.7, 1.1])
    inertia_tensor = np.array(mass_props.inertia_tensor, dtype=float)
    KE_expected = 0.5 * np.einsum("i,ij,j->", w, inertia_tensor, w)

    assert sol(KE) == pytest.approx(KE_expected)


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
