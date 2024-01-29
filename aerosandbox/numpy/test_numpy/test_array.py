from aerosandbox.numpy.array import *
import pytest
import aerosandbox.numpy as np
import casadi as cas


def test_array_numpy_equivalency_1D():
    inputs = [
        1,
        2
    ]

    a = array(inputs)
    a_np = np.array(inputs)

    assert np.all(a == a_np)


def test_array_numpy_equivalency_2D():
    inputs = [
        [1, 2],
        [3, 4]
    ]

    a = array(inputs)
    a_np = np.array(inputs)

    assert np.all(a == a_np)


def test_array_casadi_1D_shape():
    a = array([cas.DM(1), cas.DM(2)])
    assert length(a) == 2


def test_can_convert_DM_to_ndarray():
    c = cas.DM([1, 2, 3])
    n = np.array(c)

    assert np.all(n == np.array([1, 2, 3]))


def test_length():
    assert length(5) == 1
    assert length(5.) == 1
    assert length([1, 2, 3]) == 3

    assert length(np.array(5)) == 1
    assert length(np.array([5])) == 1
    assert length(np.array([1, 2, 3])) == 3
    assert length(np.ones((3, 2))) == 3

    assert length(cas.MX(np.ones(5))) == 5


def test_concatenate():
    n = np.arange(10)
    c = cas.DM(n)

    assert concatenate((n, n)).shape == (20,)
    assert concatenate((n, c)).shape == (20, 1)
    assert concatenate((c, n)).shape == (20, 1)
    assert concatenate((c, c)).shape == (20, 1)

    assert concatenate((n, n, n)).shape == (30,)
    assert concatenate((c, c, c)).shape == (30, 1)


def test_stack():
    n = np.arange(10)
    c = cas.DM(n)

    assert stack((n, n)).shape == (2, 10)
    assert stack((n, n), axis=-1).shape == (10, 2)
    assert stack((n, c)).shape == (2, 10)
    assert stack((n, c), axis=-1).shape == (10, 2)
    assert stack((c, c)).shape == (2, 10)
    assert stack((c, c), axis=-1).shape == (10, 2)

    with pytest.raises(Exception):
        assert stack((n, n), axis=2)

    with pytest.raises(Exception):
        stack((c, c), axis=2)


def test_diag_onp():
    # Test on 1D array
    a = np.array([1, 2, 3])
    assert np.all(np.diag(a) == np.diag(a, k=0))
    assert np.all(
        np.diag(a, k=1) == np.array([
            [0, 1, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 3],
            [0, 0, 0, 0]
        ])
    )
    assert np.all(
        np.diag(a, k=-1) == np.array([
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 3, 0]
        ])
    )

    # Test on 2D square array
    b = np.array([[1, 2], [3, 4]])
    assert np.all(np.diag(b) == np.array([1, 4]))
    assert np.all(np.diag(b, k=1) == np.array([2]))
    assert np.all(np.diag(b, k=-1) == np.array([3]))

    # Test on non-square 2D array
    c = np.array([[1, 2, 3], [4, 5, 6]])
    assert np.all(np.diag(c) == np.array([1, 5]))
    assert np.all(np.diag(c, k=1) == np.array([2, 6]))
    assert np.all(np.diag(c, k=-1) == np.array([4]))


def test_diag_casadi():
    # Test on 1D array
    a = cas.SX(np.array([1, 2, 3]))
    assert np.all(np.diag(a) == np.diag(a, k=0))
    assert np.all(
        np.diag(a, k=1) == np.array([
            [0, 1, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 3],
            [0, 0, 0, 0]
        ])
    )
    assert np.all(
        np.diag(a, k=-1) == np.array([
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 3, 0]
        ])
    )

    # Test on 2D square array
    b = cas.SX(np.array([[1, 2], [3, 4]]))
    assert np.all(np.diag(b) == np.array([1, 4]))
    assert np.all(np.diag(b, k=1) == np.array([2]))
    assert np.all(np.diag(b, k=-1) == np.array([3]))

    # # Test on non-square 2D array
    # c = cas.SX(np.array([[1, 2, 3], [4, 5, 6]]))
    # assert np.all(np.diag(c) == np.array([1, 5]))
    # assert np.all(np.diag(c, k=1) == np.array([2, 6]))
    # assert np.all(np.diag(c, k=-1) == np.array([4]))

def test_roll_onp():
    # Test on 1D array
    a = np.arange(1, 101)  # Large array
    b = np.concatenate([np.arange(91, 101), np.arange(1, 91)])

    assert np.all(np.roll(a, 10) == b)

    # Test negative shift on 1D array
    assert np.all(np.roll(a, -10) == np.roll(a, 90))

    # Test shift larger than array size
    assert np.all(np.roll(a, 110) == np.roll(a, 10))


def test_roll_casadi():
    # Test on 1D array
    a_np = np.arange(1, 101)
    a = cas.SX(a_np)
    b = cas.SX(np.concatenate([np.arange(91, 101), np.arange(1, 91)]))

    assert np.all(cas.DM(np.roll(a, 10)) == b)

    # Test negative shift on 1D array
    assert np.all(cas.DM(np.roll(a, -10)) == cas.DM(np.roll(a, 90)))

    # Test shift larger than array size
    assert np.all(cas.DM(np.roll(a, 110)) == cas.DM(np.roll(a, 10)))


def test_roll_casadi_2d():
    # Test on 2D array
    a_np = np.reshape(np.arange(1, 101), (10, 10))  # 2D array
    a = cas.SX(a_np)

    # Shift along axis 1
    assert np.all(cas.DM(np.roll(a, 2, axis=1)) == np.roll(a_np, 2, axis=1))

    # Shift along axis 0
    assert np.all(cas.DM(np.roll(a, 2, axis=0)) == np.roll(a_np, 2, axis=0))

    # Shift along both axes
    assert np.all(cas.DM(np.roll(a, (2, 3), axis=(0, 1))) == np.roll(a_np, (2, 3), axis=(0, 1)))

    # Test on non-square 2D array
    a_np = np.reshape(np.arange(1, 201), (10, 20))  # non-square 2D array
    a = cas.SX(a_np)

    # Shift along axis 1
    assert np.all(cas.DM(np.roll(a, 2, axis=1)) == np.roll(a_np, 2, axis=1))

    # Shift along axis 0
    assert np.all(cas.DM(np.roll(a, 2, axis=0)) == np.roll(a_np, 2, axis=0))

    # Shift along both axes
    assert np.all(cas.DM(np.roll(a, (2, 3), axis=(0, 1))) == np.roll(a_np, (2, 3), axis=(0, 1)))


def test_max():
    a = cas.SX([1, 2, 3])
    b = [1, 2, 3]

    assert int(np.max(a)) == int(np.max(b))


def test_min():
    a = cas.SX([1, 2, 3])
    b = [1, 2, 3]

    assert int(np.min(a)) == int(np.min(b))


def test_reshape_1D():
    a = np.array([1, 2, 3, 4, 5, 6])
    b = cas.DM(a)

    assert b.shape == (len(a), 1)

    test_inputs = [
        -1,
        (3, 2),
        (2, 3),
        (6, 1),
        (1, 6),
        (-1),
        (6, -1),
        (-1, 6),
    ]

    for i in test_inputs:
        ra = np.reshape(a, i)
        rb = np.reshape(b, i)
        if len(ra.shape) == 1:
            ra = ra.reshape(-1, 1)

        assert np.all(ra == rb)
        assert ra.shape == rb.shape


def test_reshape_2D_vec_tall():
    a = np.array([1, 2, 3, 4, 5, 6]).reshape((6, 1))
    b = cas.DM(a)

    assert b.shape == (len(a), 1)

    test_inputs = [
        -1,
        (3, 2),
        (2, 3),
        (6, 1),
        (1, 6),
        (-1),
        (6, -1),
        (-1, 6),
    ]

    for i in test_inputs:
        ra = np.reshape(a, i)
        rb = np.reshape(b, i)
        if len(ra.shape) == 1:
            ra = ra.reshape(-1, 1)

        assert np.all(ra == rb)
        assert ra.shape == rb.shape


def test_reshape_2D_vec_wide():
    a = np.array([1, 2, 3, 4, 5, 6]).reshape((1, 6))
    b = cas.DM(a)

    assert b.shape == (1, 6)

    test_inputs = [
        -1,
        (3, 2),
        (2, 3),
        (6, 1),
        (1, 6),
        (-1),
        (6, -1),
        (-1, 6),
    ]

    for i in test_inputs:
        ra = np.reshape(a, i)
        rb = np.reshape(b, i)
        if len(ra.shape) == 1:
            ra = ra.reshape(-1, 1)

        assert np.all(ra == rb)
        assert ra.shape == rb.shape


def test_reshape_2D():
    a_np = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
    ])
    a_cas = cas.DM(a_np)

    test_inputs = [
        -1,
        (4, 3),
        (3, 4),
        (12, 1),
        (1, 12),
        (-1),
        (12, -1),
        (-1, 12),
    ]

    for i in test_inputs:
        res_np = np.reshape(a_np, i)
        res_cas = np.reshape(a_cas, i)
        if len(res_np.shape) == 1:
            res_np = res_np.reshape(-1, 1)

        assert np.all(res_np == res_cas)
        assert res_np.shape == res_cas.shape

    for i in test_inputs:
        res_np = np.reshape(a_np, i, order="F")
        res_cas = np.reshape(a_cas, i, order="F")
        if len(res_np.shape) == 1:
            res_np = res_np.reshape(-1, 1)

        assert np.all(res_np == res_cas)
        assert res_np.shape == res_cas.shape


def test_assert_equal_shape():
    a = np.array([1, 2, 3])
    b = cas.DM(a)

    np.assert_equal_shape([
        a,
        a
    ])
    np.assert_equal_shape({
        "thing1": a,
        "thing2": a,
    })
    with pytest.raises(ValueError):
        np.assert_equal_shape([
            np.array([1, 2, 3]),
            np.array([1, 2, 3, 4])
        ])
        np.assert_equal_shape({
            "thing1": np.array([1, 2, 3]),
            "thing2": np.array([1, 2, 3, 4])
        })
    np.assert_equal_shape([
        2,
        3,
        4
    ])


if __name__ == '__main__':
    # # Test on 1D array
    # a_np = np.arange(1, 101)
    # a = cas.SX(a_np)
    # b = cas.SX(np.concatenate([np.arange(91, 101), np.arange(1, 91)]))
    #
    # s1 = np.roll(a, -10, axis=0)
    # s2 = np.roll(a, 90, axis=0)
    #
    # assert np.all(cas.DM(s1) == cas.DM(s2))

    pytest.main()
