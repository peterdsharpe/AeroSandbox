import aerosandbox as asb
import aerosandbox.numpy as np
import pytest


def test_airfoil_with_TE_gap():
    a = asb.AirfoilInviscid(
        airfoil=asb.Airfoil("naca4408").repanel(100),
        op_point=asb.OperatingPoint(
            velocity=1,
            alpha=5
        )
    )
    assert a.Cl == pytest.approx(1.0754, abs=0.01)  # From XFoil


def test_airfoil_without_TE_gap():
    a = asb.AirfoilInviscid(
        airfoil=asb.Airfoil("e423").repanel(100),
        op_point=asb.OperatingPoint(
            velocity=1,
            alpha=5
        )
    )
    assert a.Cl == pytest.approx(1.9304, abs=0.01)  # From XFoil


def test_airfoil_multielement():
    a = asb.AirfoilInviscid(
        airfoil=[
            asb.Airfoil("e423")
                .repanel(n_points_per_side=50),
            asb.Airfoil("naca6408")
                .repanel(n_points_per_side=25)
                .scale(0.4, 0.4)
                .rotate(np.radians(-20))
                .translate(0.9, -0.05),
        ],
        op_point=asb.OperatingPoint(
            velocity=1,
            alpha=5
        )
    )


def test_airfoil_ground_effect():
    a = asb.AirfoilInviscid(
        airfoil=asb.Airfoil("naca4408").repanel(100).translate(0, 0.2),
        op_point=asb.OperatingPoint(
            velocity=1,
            alpha=0
        ),
        ground_effect=True
    )
    assert a.calculate_velocity(0, 0)[1] == pytest.approx(0)


if __name__ == '__main__':
    # pass
    pytest.main()
