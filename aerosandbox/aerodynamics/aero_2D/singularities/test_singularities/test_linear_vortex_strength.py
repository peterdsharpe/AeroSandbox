from aerosandbox.aerodynamics.aero_2D.singularities import *
import aerosandbox.numpy as np
from numpy import pi
import pytest


def test_calculate_induced_velocity_panel_coordinates():
    X, Y = np.meshgrid(
        np.linspace(-1, 2, 50),
        np.linspace(-1, 1, 50),
        indexing='ij',
    )
    X = X.flatten()
    Y = Y.flatten()

    U, V = calculate_induced_velocity_line_singularities(
        x_field=X,
        y_field=Y,
        x_panels=np.array([-0.5, 1.5]),
        y_panels=np.array([0, 0]),
        gamma=np.array([1, 1]),
        sigma=np.array([1, 1]),
    )


def test_vortex_limit():
    eps = 1e-6
    u, v = calculate_induced_velocity_line_singularities(
        x_field=1,
        y_field=0,
        x_panels=[0, eps],
        y_panels=[0, 0],
        gamma=[1 / eps, 1 / eps],
        sigma=[0, 0],
    )
    assert u == pytest.approx(0, abs=eps ** 0.5)
    assert v == pytest.approx(-1 / (2 * pi), abs=eps ** 0.5)


def test_source_limit():
    eps = 1e-6
    u, v = calculate_induced_velocity_line_singularities(
        x_field=1,
        y_field=0,
        x_panels=[0, eps],
        y_panels=[0, 0],
        sigma=[1 / eps, 1 / eps],
        gamma=[0, 0],
    )
    assert u == pytest.approx(1 / (2 * pi), abs=eps ** 0.5)
    assert v == pytest.approx(0, abs=eps ** 0.5)


def test_zero_length_case():
    u, v = calculate_induced_velocity_line_singularities(
        x_field=1,
        y_field=1,
        x_panels=[0, 0],
        y_panels=[0, 0],
        sigma=[1, 1],
        gamma=[1, 1],
    )

    assert u == pytest.approx(0)
    assert v == pytest.approx(0)


if __name__ == '__main__':
    test_zero_length_case()
    pytest.main()
