import pytest
import aerosandbox as asb
import aerosandbox.numpy as np


def test_uniform_forward_difference_first_degree():
    assert np.finite_difference_coefficients(
        x=np.arange(2),
        x0=0,
        derivative_degree=1
    ) == pytest.approx(
        np.array([
            -1, 1
        ])
    )
    assert np.finite_difference_coefficients(
        x=np.arange(9),
        x0=0,
        derivative_degree=1
    ) == pytest.approx(
        np.array([
            -761 / 280,
            8,
            -14,
            56 / 3,
            -35 / 2,
            56 / 5,
            -14 / 3,
            8 / 7,
            -1 / 8
        ])
    )


def test_uniform_forward_difference_higher_order():
    assert np.finite_difference_coefficients(
        x=np.arange(5),
        x0=0,
        derivative_degree=3
    ) == pytest.approx(
        np.array([
            -5 / 2,
            9,
            -12,
            7,
            -3 / 2
        ])
    )


def test_uniform_central_difference():
    assert np.finite_difference_coefficients(
        x=[-1, 0, 1],
        x0=0,
        derivative_degree=1
    ) == pytest.approx(
        np.array([
            -0.5,
            0,
            0.5
        ])
    )
    assert np.finite_difference_coefficients(
        x=[-1, 0, 1],
        x0=0,
        derivative_degree=2
    ) == pytest.approx(
        np.array([
            1,
            -2,
            1
        ])
    )
    assert np.finite_difference_coefficients(
        x=[-2, -1, 0, 1, 2],
        x0=0,
        derivative_degree=2
    ) == pytest.approx(
        np.array([
            -1 / 12,
            4 / 3,
            -5 / 2,
            4 / 3,
            -1 / 12
        ])
    )


def test_nonuniform_difference():
    assert np.finite_difference_coefficients(
        x=[-1, 2],
        x0=0,
        derivative_degree=1
    ) == pytest.approx(
        np.array([
            -1 / 3,
            1 / 3
        ])
    )


if __name__ == '__main__':
    pytest.main()
