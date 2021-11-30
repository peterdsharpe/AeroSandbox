import aerosandbox.numpy as np
import casadi as cas
import pytest


def test_where_numpy():
    a = np.ones(4)
    b = 2 * np.ones(4)

    c = np.where(
        np.array([True, False, True, False]),
        a,
        b
    )

    assert np.all(
        c == np.array([1, 2, 1, 2])
    )


def test_where_casadi():
    a = cas.GenDM_ones(4)
    b = 2 * cas.GenDM_ones(4)

    c = np.where(
        cas.DM([1, 0, 1, 0]),
        a,
        b
    )

    assert np.all(
        c == cas.DM([1, 2, 1, 2])
    )

# def test_if_else_mixed(): # TODO write this


if __name__ == '__main__':
    pytest.main()
