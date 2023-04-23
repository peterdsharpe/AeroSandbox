import aerosandbox as asb
import aerosandbox.numpy as np
import pytest
from pprint import pprint



def test_fuselage_static_stability_derivatives():
    fuselage = asb.Fuselage(
        xsecs=[
            asb.FuselageXSec(
                xyz_c=[xi, 0, 0],
                radius=asb.Airfoil("naca0010").local_thickness(0.8 * xi)
            )
            for xi in np.cosspace(0, 1, 20)
        ],
    )

    s_ref = 10
    c_ref = 100
    b_ref = 1000

    airplane = asb.Airplane(
        fuselages=[fuselage],
        s_ref=s_ref,
        c_ref=c_ref,
        b_ref=b_ref
    )

    aero = asb.AeroBuildup(
        airplane=airplane,
        op_point=asb.OperatingPoint(
            velocity=10,
            alpha=1,
            beta=1
        ),
        xyz_ref=[0.5, 0, 0]
    ).run_with_stability_derivatives(
        alpha=True,
        beta=True,
        p=False,
        q=False,
        r=False
    )

    # pprint(aero)

    slender_body_theory_aero = dict(
        Cma=2 * fuselage.volume() / s_ref / c_ref,
        Cnb=-2 * fuselage.volume() / s_ref / b_ref
    )

    # pprint(sbt)

    assert aero["Cma"] == pytest.approx(
        slender_body_theory_aero["Cma"],
        rel=3
    )
    assert aero["Cnb"] == pytest.approx(
        slender_body_theory_aero["Cnb"],
        rel=3
    )


if __name__ == '__main__':
    test_fuselage_static_stability_derivatives()
    # pytest.main()
