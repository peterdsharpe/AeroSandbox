from aerosandbox.geometry.airfoil import *
import pytest

def test_init():
    a = Airfoil("naca4412")
    assert a.TE_angle() == pytest.approx(14.766578406372423)

if __name__ == '__main__':
    test_init()