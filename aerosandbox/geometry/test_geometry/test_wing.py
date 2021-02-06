from aerosandbox import *
import pytest


@pytest.fixture()
def w():
    wing = Wing(
        name="MyWing",
        xyz_le=np.array([1, 2, 3]),
        xsecs=[
            WingXSec(
                xyz_le=np.array([1, 1, 0]),
                chord=0.5,
                twist_angle=5,
                control_surface_is_symmetric=True
            ),
            WingXSec(
                xyz_le=np.array([2, 2, 0]),
                chord=0.5,
                twist_angle=5,
                control_surface_is_symmetric=True
            )
        ],
        symmetric=True
    )
    return wing


def test_span(w):
    assert w.span() == pytest.approx(2)


def test_area(w):
    assert w.area() == pytest.approx(1)


def test_aspect_ratio(w):
    assert w.aspect_ratio() == pytest.approx(4)


def test_is_entirely_symmetric(w):
    assert w.is_entirely_symmetric()


def test_mean_geometric_chord(w):
    assert w.mean_geometric_chord() == pytest.approx(0.5)


def test_mean_aerodynamic_chord(w):
    assert w.mean_aerodynamic_chord() == pytest.approx(0.5)


def test_mean_twist_angle(w):
    assert w.mean_twist_angle() == pytest.approx(5)


def test_mean_sweep_angle(w):
    assert w.mean_sweep_angle() == pytest.approx(45)


def test_aerodynamic_center(w):
    ac = w.aerodynamic_center()

    assert ac[0] == pytest.approx(1 + 1.5 + 1 / 8, abs=2e-2)
    assert ac[1] == pytest.approx(0)
    assert ac[2] == pytest.approx(3, abs=2e-2)


if __name__ == '__main__':
    pytest.main()
