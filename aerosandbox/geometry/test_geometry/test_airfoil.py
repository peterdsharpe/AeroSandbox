from aerosandbox.geometry.airfoil import *
import pytest


@pytest.fixture
def naca4412():
    return Airfoil("naca4412")


@pytest.fixture
def e216():
    a = Airfoil("e216")
    assert a.n_points() == 61
    return a


def test_fake_airfoil():
    a = Airfoil("dae12")
    assert a.coordinates is None
    assert a.n_points() == 0


def test_TE_angle(naca4412):
    assert naca4412.TE_angle() == pytest.approx(14.74635802332286, abs=1)


def test_local_thickness(e216):
    assert e216.local_thickness(0.5) == pytest.approx(0.08730287761717835)


def test_LE_index(e216):
    assert e216.LE_index() == 32


def test_repanel(naca4412):
    naca4412 = naca4412.repanel(n_points_per_side=300)
    assert naca4412.n_points() == 599


def test_containts_points(naca4412):
    assert naca4412.contains_points(
        x=0.5, y=0
    ) == True
    assert np.all(naca4412.contains_points(
        x=np.array([0.5, 0.5]),
        y=np.array([0, -0.1])
    ) == np.array([True, False]))
    shape = (1, 2, 3, 4)
    x_points = np.random.randn(*shape)
    y_points = np.random.randn(*shape)
    contains = naca4412.contains_points(x_points, y_points)
    assert shape == contains.shape


if __name__ == '__main__':
    pytest.main()
