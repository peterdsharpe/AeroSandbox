from aerosandbox.geometry.airfoil.airfoil_families import *
import pytest


def test_get_NACA_coordinates():
    coords = get_NACA_coordinates(
        name='naca4408',
        n_points_per_side=100
    )
    assert len(coords) == 199


def test_get_UIUC_coordinates():
    coords = get_UIUC_coordinates(
        name="dae11",
    )
    assert len(coords) != 0


if __name__ == '__main__':
    pytest.main()
