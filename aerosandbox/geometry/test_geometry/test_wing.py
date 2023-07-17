from aerosandbox import *
import pytest


def w() -> Wing:
    wing = Wing(
        name="MyWing",
        xsecs=[
            WingXSec(
                xyz_le=np.array([1, 1, 0]),
                chord=0.5,
                twist=5,
                airfoil=Airfoil("mh60"),
                control_surfaces=[
                    ControlSurface(
                        symmetric=True,
                    )
                ]
            ),
            WingXSec(
                xyz_le=np.array([2, 2, 0]),
                chord=0.5,
                twist=5,
                airfoil=Airfoil("mh60"),
                control_surfaces=[
                    ControlSurface(
                        symmetric=True
                    )
                ]
            )
        ],
        symmetric=True
    ).translate(np.array([1, 2, 3]))
    return wing


def test_span():
    assert w().span() == pytest.approx(2)


def test_area():
    assert w().area() == pytest.approx(1)


def test_aspect_ratio():
    assert w().aspect_ratio() == pytest.approx(4)


def test_is_entirely_symmetric():
    assert w().is_entirely_symmetric()


def test_mean_geometric_chord():
    assert w().mean_geometric_chord() == pytest.approx(0.5)


def test_mean_aerodynamic_chord():
    assert w().mean_aerodynamic_chord() == pytest.approx(0.5)


def test_mean_twist_angle():
    assert w().mean_twist_angle() == pytest.approx(5)


def test_mean_sweep_angle():
    assert w().mean_sweep_angle() == pytest.approx(45)


def test_aerodynamic_center():
    ac = w().aerodynamic_center()

    assert ac[0] == pytest.approx(1 + 1.5 + 1 / 8, abs=2e-2)
    assert ac[1] == pytest.approx(0)
    assert ac[2] == pytest.approx(3, abs=2e-2)


if __name__ == '__main__':
    pytest.main()
