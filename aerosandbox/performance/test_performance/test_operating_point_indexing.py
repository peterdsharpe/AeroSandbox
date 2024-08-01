import aerosandbox as asb
import aerosandbox.numpy as np


def test_op_indexing():

    op = asb.OperatingPoint(
        atmosphere=asb.Atmosphere(altitude=np.linspace(0, 100)),
        velocity=np.linspace(30, 70),
        alpha=np.linspace(-5, 5),
        beta=0,
    )
    o0 = op[0]
    o1 = op[1]
    on1 = op[-1]
    on2 = op[-2]
    oslice = op[2:5]

    assert o0.atmosphere == op.atmosphere[0]
    assert o0.alpha == op.alpha[0]
    assert o0.velocity == op.velocity[0]

    assert o1.atmosphere == op.atmosphere[1]
    assert o1.alpha == op.alpha[1]
    assert o1.velocity == op.velocity[1]

    assert on1.atmosphere == op.atmosphere[-1]
    assert on1.alpha == op.alpha[-1]
    assert on1.velocity == op.velocity[-1]

    assert on2.atmosphere == op.atmosphere[-2]
    assert on2.alpha == op.alpha[-2]
    assert on2.velocity == op.velocity[-2]

    assert oslice.atmosphere == op.atmosphere[2:5]
    assert np.all(oslice.alpha == op.alpha[2:5])
    assert np.all(oslice.velocity == op.velocity[2:5])

    # # Test indexing of a simple Dynamics object
    # dyn = asb.DynamicsPointMass1DHorizontal(
    #     mass_props=asb.MassProperties(
    #         mass=1
    #     ),
    #     x_e=np.arange(10) ** 2,
    #     u_e=2 * np.arange(10),
    # )
    #
    # d0 = dyn[0]
    # d1 = dyn[1]
    # dn1 = dyn[-1]
    # dn2 = dyn[-2]
    # dslice = dyn[2:5]
    #
    # assert d0.x_e == 0
    # assert d0.u_e == 0
    # assert d1.x_e == 1
    # assert d1.u_e == 2
    # assert dn1.x_e == 81
    # assert dn1.u_e == 18
    # assert dn2.x_e == 64
    # assert dn2.u_e == 16
    # assert all(dslice.x_e == [4, 9, 16])
    # assert all(dslice.u_e == [4, 6, 8])


if __name__ == "__main__":
    test_op_indexing()
