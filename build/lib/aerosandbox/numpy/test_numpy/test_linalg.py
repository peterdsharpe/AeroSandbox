import pytest
import aerosandbox.numpy as np
import casadi as cas


def test_norm_vector():
    a = np.array([1, 2, 3])
    cas_a = cas.DM(a)

    assert np.linalg.norm(a) == np.linalg.norm(cas_a)


def test_norm_2D():
    a = np.arange(9).reshape(3, 3)
    cas_a = cas.DM(a)

    assert np.linalg.norm(cas_a) == np.linalg.norm(a)

    assert np.all(
        np.linalg.norm(cas_a, axis=0) ==
        np.linalg.norm(a, axis=0)
    )

    assert np.all(
        np.linalg.norm(cas_a, axis=1) ==
        np.linalg.norm(a, axis=1)
    )


if __name__ == '__main__':
    pytest.main()
