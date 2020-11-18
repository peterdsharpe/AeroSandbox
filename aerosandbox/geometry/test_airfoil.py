from aerosandbox.geometry.airfoil import *
import pytest

@pytest.fixture
def naca4412():
    return Airfoil("naca4412")

@pytest.fixture
def e216():
    a = Airfoil("e216")
    assert len(a.coordinates) == 61
    return a

def test_TE_angle(naca4412):
    assert naca4412.TE_angle() == pytest.approx(14.766578406372423)

def test_local_thickness(e216):
    assert e216.local_thickness(0.5) == pytest.approx(0.08730287761717835)

def LE_index_test(e216):
    assert e216.LE_index() == 32

def repanel_test(naca4412):
    naca4412.repanel(n_points_per_side=200, inplace=True)
    # assert

if __name__ == '__main__':
    pytest.main()