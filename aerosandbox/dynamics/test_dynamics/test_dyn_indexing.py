import aerosandbox as asb
import aerosandbox.numpy as np


def test_dyn_indexing():

    # Test indexing of a simple Dynamics object
    dyn = asb.DynamicsPointMass1DHorizontal(
        mass_props=asb.MassProperties(mass=1),
        x_e=np.arange(10) ** 2,
        u_e=2 * np.arange(10),
    )

    d0 = dyn[0]
    d1 = dyn[1]
    dn1 = dyn[-1]
    dn2 = dyn[-2]
    dslice = dyn[2:5]

    assert d0.x_e == 0
    assert d0.u_e == 0
    assert d1.x_e == 1
    assert d1.u_e == 2
    assert dn1.x_e == 81
    assert dn1.u_e == 18
    assert dn2.x_e == 64
    assert dn2.u_e == 16
    assert all(dslice.x_e == [4, 9, 16])
    assert all(dslice.u_e == [4, 6, 8])


if __name__ == "__main__":
    test_dyn_indexing()
