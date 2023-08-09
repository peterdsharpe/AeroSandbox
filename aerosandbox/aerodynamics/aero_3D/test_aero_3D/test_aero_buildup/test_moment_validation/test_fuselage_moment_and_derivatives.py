import aerosandbox as asb
import aerosandbox.numpy as np
from typing import Type
import pytest

rtol = 0.10


def make_fuselage(
        alpha_geometric=0.,
        beta_geometric=0.
) -> asb.Fuselage:
    x = [0, 0.2, 0.8, 1]

    return asb.Fuselage(
        xsecs=[
            asb.FuselageXSec(
                xyz_c=[
                    xi * np.cosd(alpha_geometric) * np.cosd(beta_geometric),
                    xi * np.sind(beta_geometric),
                    xi * np.sind(-alpha_geometric)
                ],
                radius=np.where(
                    xi < 0.2,
                    0.25 * xi / 0.2,
                    np.where(
                        xi < 0.8,
                        0.25,
                        0.25 * (1 - xi) / 0.2
                    )
                )
            )
            for xi in x
        ]
    ).subdivide_sections(10)


def test_derivatives_at_zero_geometric_angles(
        AeroAnalysis: Type = asb.AeroBuildup,
):

    fuselage = make_fuselage(
        alpha_geometric=0.,
        beta_geometric=0.
    )

    airplane = asb.Airplane(
        fuselages=[fuselage],
        s_ref=np.pi,
        c_ref=1,
        b_ref=0.5,
    )

    analysis = AeroAnalysis(
        airplane=airplane,
        op_point=asb.OperatingPoint(
            velocity=100,
            alpha=0,
            beta=0,
        ),
        xyz_ref=np.array([0, 0, 0]),
    )

    try:
        aero = analysis.run_with_stability_derivatives()
    except AttributeError:
        aero = analysis.run()

    print(f"Aerodynamic coefficients with {AeroAnalysis.__name__}:")
    for key in ["Cm", "Cma", "Cn", "Cnb"]:
        print(f"{key.rjust(10)}: {aero[key]:20.4f}")

    assert aero["Cm"] == pytest.approx(0, abs=1e-3)
    assert aero["Cn"] == pytest.approx(0, abs=1e-3)

    assert aero["Cma"] == pytest.approx(
        2 * fuselage.volume() / airplane.s_ref / airplane.c_ref,
        rel=rtol
    )
    assert aero["Cnb"] == pytest.approx(
        -2 * fuselage.volume() / airplane.s_ref / airplane.b_ref,
        rel=rtol
    )


def test_derivatives_at_constant_offset(
        AeroAnalysis: Type = asb.AeroBuildup,
):

    fuselage = make_fuselage(
        alpha_geometric=5.,
        beta_geometric=10.
    )

    airplane = asb.Airplane(
        fuselages=[fuselage],
        s_ref=np.pi,
        c_ref=1.,
        b_ref=0.5,
    )

    analysis = AeroAnalysis(
        airplane=airplane,
        op_point=asb.OperatingPoint(
            velocity=100.,
            alpha=-5.,
            beta=-10.,
        ),
        xyz_ref=np.array([0, 0, 0]),
    )

    try:
        aero = analysis.run_with_stability_derivatives()
    except AttributeError:
        aero = analysis.run()

    print(f"Aerodynamic coefficients with {AeroAnalysis.__name__}:")
    for key in ["Cm", "Cma", "Cn", "Cnb"]:
        print(f"{key.rjust(10)}: {aero[key]:20.4f}")

    # assert aero["Cm"] == pytest.approx(0, abs=1e-3)
    # assert aero["Cn"] == pytest.approx(0, abs=1e-3)

    assert aero["Cma"] == pytest.approx(
        2 * fuselage.volume() / airplane.s_ref / airplane.c_ref,
        rel=rtol
    )
    assert aero["Cnb"] == pytest.approx(
        -2 * fuselage.volume() / airplane.s_ref / airplane.b_ref,
        rel=rtol
    )


def test_derivatives_at_incidence(
        AeroAnalysis: Type = asb.AeroBuildup,
):

    fuselage = make_fuselage(
        alpha_geometric=0.,
        beta_geometric=0.
    )

    airplane = asb.Airplane(
        fuselages=[fuselage],
        s_ref=np.pi,
        c_ref=1.,
        b_ref=0.5,
    )

    analysis = AeroAnalysis(
        airplane=airplane,
        op_point=asb.OperatingPoint(
            velocity=100.,
            alpha=5.,
            beta=5.,
        ),
        xyz_ref=np.array([0, 0, 0]),
    )

    try:
        aero = analysis.run_with_stability_derivatives()
    except AttributeError:
        aero = analysis.run()

    print(f"Aerodynamic coefficients with {AeroAnalysis.__name__}:")
    for key in ["Cm", "Cma", "Cn", "Cnb", "CL", "CY", "CD"]:
        print(f"{key.rjust(10)}: {aero[key]:20.4f}")

    assert aero["Cm"] == pytest.approx(
        2 * fuselage.volume() / airplane.s_ref / airplane.c_ref * np.radians(5),
        rel=rtol
    )
    assert aero["Cn"] == pytest.approx(
        -2 * fuselage.volume() / airplane.s_ref / airplane.b_ref * np.radians(5),
        rel=rtol
    )

    assert aero["Cma"] == pytest.approx(
        2 * fuselage.volume() / airplane.s_ref / airplane.c_ref,
        rel=rtol
    )
    assert aero["Cnb"] == pytest.approx(
        -2 * fuselage.volume() / airplane.s_ref / airplane.b_ref,
        rel=rtol
    )


def test_cambered_fuselage(
        AeroAnalysis: Type = asb.AeroBuildup,
):
    fuselage = asb.Fuselage(
        xsecs=[
            asb.FuselageXSec(
                xyz_c=[
                    xi,
                    0,
                    xi * (1 - xi) * 4 * 0.15
                ],
                radius=xi ** 0.5 * (1 - xi) * 0.2
            )
            for xi in np.sinspace(0, 1, 20)
        ]
    )
    # fuse.draw_three_view()

    airplane = asb.Airplane(
        fuselages=[fuselage],
        s_ref=1,
        c_ref=1,
        b_ref=1,
    )

    analysis = AeroAnalysis(
        airplane=airplane,
        op_point=asb.OperatingPoint(
            velocity=100,
            alpha=5,
            beta=5,
        ),
        xyz_ref=np.array([0.5, 0, 0]),
    )

    try:
        aero = analysis.run_with_stability_derivatives()
    except AttributeError:
        aero = analysis.run()

    print(f"Aerodynamic coefficients with {AeroAnalysis.__name__}:")
    for key in ["Cm", "Cma", "Cn", "Cnb"]:
        print(f"{key.rjust(10)}: {aero[key]:20.4f}")

    # assert aero["Cm"] == pytest.approx(0, abs=1e-3)
    # assert aero["Cn"] == pytest.approx(0, abs=1e-3)

    assert aero["Cma"] == pytest.approx(
        2 * fuselage.volume() / airplane.s_ref / airplane.c_ref,
        rel=0.5 # this is not-strictly-speaking exact, hence the large tolerance
    )
    assert aero["Cnb"] == pytest.approx(
        -2 * fuselage.volume() / airplane.s_ref / airplane.b_ref,
        rel=0.5 # this is not-strictly-speaking exact, hence the large tolerance
    )


if __name__ == '__main__':
    pass
    # asb.Airplane(fuselages=[make_fuselage()]).draw_three_view()
    # # test_derivatives_at_zero_geometric_angles(asb.AVL)
    # test_derivatives_at_zero_geometric_angles(asb.AeroBuildup)
    # # test_derivatives_at_constant_offset(asb.AVL)
    # test_derivatives_at_constant_offset(asb.AeroBuildup)
    # # test_derivatives_at_incidence(asb.AVL)
    # test_derivatives_at_incidence(asb.AeroBuildup)
    # # test_cambered_fuselage(asb.AVL)
    # test_cambered_fuselage(asb.AeroBuildup)
