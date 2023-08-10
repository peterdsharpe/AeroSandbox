import aerosandbox as asb
import aerosandbox.numpy as np
import pytest

rtol = 0.10

def make_fuselage(
        diameter = 0.1,
        length = 10,
        alpha_geometric=0.,
        beta_geometric=0.,
        N=101,
):
    return asb.Fuselage(
        xsecs=[
            asb.FuselageXSec(
                xyz_c=[
                    xi * np.cosd(alpha_geometric) * np.cosd(beta_geometric),
                    xi * np.sind(beta_geometric),
                    xi * np.sind(-alpha_geometric)
                ],
                radius=diameter / 2
            )
            for xi in [0, length]
        ]
    ).subdivide_sections(N)

def test_90_deg_crossflow():
    diameter = 0.1
    length = 10

    fuselage = make_fuselage(
        diameter=diameter,
        length=length,
        alpha_geometric=0
    )

    airplane = asb.Airplane(
        fuselages=[fuselage],
        s_ref=diameter * length,
        c_ref=length,
        b_ref=diameter,
    )

    op_point = asb.OperatingPoint(
            velocity=50,
            alpha=90,
            beta=0,
        )

    aero = asb.AeroBuildup(
        airplane=airplane,
        op_point=op_point,
        xyz_ref=np.array([length / 2, 0, 0])
    ).run()

    print(aero)

    from aerosandbox.library.aerodynamics.viscous import Cd_cylinder

    CD_expected = Cd_cylinder(
        Re_D=op_point.reynolds(diameter),
        mach=op_point.mach()
    )
    print(CD_expected)

    assert aero["CD"] == pytest.approx(
        CD_expected, rel=0.1
    )


if __name__ == '__main__':
    pass
    # make_fuselage().draw_three_view()
    test_90_deg_crossflow()