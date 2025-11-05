import aerosandbox.numpy as np
from aerosandbox.numpy.linalg import (
    inner,
    outer,
    solve,
    inv,
    pinv,
    det,
    norm,
    inv_symmetric_3x3,
)
import pytest
import casadi as cas


def test_inner_product_numpy():
    """Test inner product with NumPy arrays."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])

    result = inner(x, y)
    expected = 1 * 4 + 2 * 5 + 3 * 6

    assert np.isclose(result, expected)


def test_inner_product_manual():
    """Test inner product with manual=True flag."""
    x = [1, 2, 3]
    y = [4, 5, 6]

    result = inner(x, y, manual=True)
    expected = 1 * 4 + 2 * 5 + 3 * 6

    assert np.isclose(result, expected)


def test_inner_product_casadi():
    """Test inner product with CasADi types."""
    x = cas.DM([1, 2, 3])
    y = cas.DM([4, 5, 6])

    result = inner(x, y)
    expected = 32

    assert float(result) == expected


def test_outer_product_numpy():
    """Test outer product with NumPy arrays."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5])

    result = outer(x, y)
    expected = np.array([[4, 5], [8, 10], [12, 15]])

    assert np.allclose(result, expected)


def test_outer_product_manual():
    """Test outer product with manual=True flag."""
    x = [1, 2, 3]
    y = [4, 5]

    result = outer(x, y, manual=True)
    expected = [[4, 5], [8, 10], [12, 15]]

    assert np.allclose(result, expected)


def test_outer_product_casadi():
    """Test outer product with CasADi types."""
    x = cas.DM([1, 2, 3])
    y = cas.DM([4, 5])

    result = outer(x, y)
    expected = np.array([[4, 5], [8, 10], [12, 15]])

    assert np.allclose(np.array(result), expected)


def test_solve_linear_system_numpy():
    """Test solving a linear system Ax=b with NumPy."""
    A = np.array([[3, 1], [1, 2]], dtype=float)
    b = np.array([9, 8], dtype=float)

    x = solve(A, b)
    expected = np.array([2, 3], dtype=float)

    assert np.allclose(x, expected)


def test_solve_linear_system_casadi():
    """Test solving a linear system Ax=b with CasADi."""
    A = cas.DM([[3, 1], [1, 2]])
    b = cas.DM([9, 8])

    x = solve(A, b)
    expected = np.array([[2], [3]])

    assert np.allclose(np.array(x), expected, rtol=1e-6)


def test_inv_numpy():
    """Test matrix inversion with NumPy."""
    A = np.array([[4, 7], [2, 6]], dtype=float)

    A_inv = inv(A)
    identity = A @ A_inv

    assert np.allclose(identity, np.eye(2))


def test_inv_casadi():
    """Test matrix inversion with CasADi."""
    A = cas.DM([[4, 7], [2, 6]])

    A_inv = inv(A)
    identity = A @ A_inv

    assert np.allclose(np.array(identity), np.eye(2), rtol=1e-6)


def test_pinv_numpy():
    """Test Moore-Penrose pseudoinverse with NumPy."""
    ### Non-square matrix
    A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)

    A_pinv = pinv(A)

    ### For overdetermined system, A^+ A should be identity
    result = A_pinv @ A
    assert np.allclose(result, np.eye(2), rtol=1e-6)


def test_pinv_casadi():
    """Test Moore-Penrose pseudoinverse with CasADi."""
    A = cas.DM([[1, 2], [3, 4], [5, 6]])

    A_pinv = pinv(A)
    result = A_pinv @ A

    assert np.allclose(np.array(result), np.eye(2), rtol=1e-5)


def test_det_numpy():
    """Test determinant calculation with NumPy."""
    A = np.array([[1, 2], [3, 4]], dtype=float)

    det_A = det(A)
    expected = 1 * 4 - 2 * 3

    assert np.isclose(det_A, expected)


def test_det_casadi():
    """Test determinant calculation with CasADi."""
    A = cas.DM([[1, 2], [3, 4]])

    det_A = det(A)
    expected = -2

    assert np.isclose(float(det_A), expected)


def test_norm_l1_numpy():
    """Test L1 norm with NumPy."""
    x = np.array([3, -4, 5])

    result = norm(x, ord=1)
    expected = 3 + 4 + 5

    assert np.isclose(result, expected)


def test_norm_l2_numpy():
    """Test L2 norm with NumPy."""
    x = np.array([3, 4])

    result = norm(x, ord=2)
    expected = 5.0

    assert np.isclose(result, expected)


def test_norm_l2_casadi():
    """Test L2 norm with CasADi."""
    x = cas.DM([3, 4])

    result = norm(x, ord=2)
    expected = 5.0

    assert np.isclose(float(result), expected)


def test_norm_frobenius_numpy():
    """Test Frobenius norm with NumPy."""
    A = np.array([[1, 2], [3, 4]], dtype=float)

    result = norm(A, ord="fro")
    expected = np.sqrt(1 + 4 + 9 + 16)

    assert np.isclose(result, expected)


def test_norm_frobenius_casadi():
    """Test Frobenius norm with CasADi."""
    A = cas.DM([[1, 2], [3, 4]])

    result = norm(A, ord="fro")
    expected = np.sqrt(1 + 4 + 9 + 16)

    assert np.isclose(float(result), expected, rtol=1e-6)


def test_norm_with_axis():
    """Test norm computation along specific axis."""
    A = np.array([[3, 4], [5, 12]], dtype=float)

    ### L2 norm along axis 1 (rows)
    result = norm(A, ord=2, axis=1)
    expected = np.array([5, 13], dtype=float)

    assert np.allclose(result, expected)


def test_norm_keepdims_numpy():
    """Test norm with keepdims=True for NumPy."""
    x = np.array([[3], [4]])

    result = norm(x, ord=2, axis=0, keepdims=True)

    ### Result should maintain dimensionality
    assert result.shape == (1, 1)


def test_norm_custom_order():
    """Test norm with custom order."""
    x = np.array([1, 2, 3], dtype=float)

    result = norm(x, ord=3)
    expected = (1**3 + 2**3 + 3**3) ** (1 / 3)

    assert np.isclose(result, expected)


def test_inv_symmetric_3x3_identity():
    """Test inverse of 3x3 identity matrix."""
    ### Identity matrix has all diagonal elements = 1, off-diagonal = 0
    m11, m22, m33 = 1.0, 1.0, 1.0
    m12, m23, m13 = 0.0, 0.0, 0.0

    a11, a22, a33, a12, a23, a13 = inv_symmetric_3x3(m11, m22, m33, m12, m23, m13)

    ### Inverse of identity should be identity
    assert np.isclose(a11, 1.0)
    assert np.isclose(a22, 1.0)
    assert np.isclose(a33, 1.0)
    assert np.isclose(a12, 0.0)
    assert np.isclose(a23, 0.0)
    assert np.isclose(a13, 0.0)


def test_inv_symmetric_3x3_diagonal():
    """Test inverse of 3x3 diagonal matrix."""
    ### Diagonal matrix with [2, 3, 4] on diagonal
    m11, m22, m33 = 2.0, 3.0, 4.0
    m12, m23, m13 = 0.0, 0.0, 0.0

    a11, a22, a33, a12, a23, a13 = inv_symmetric_3x3(m11, m22, m33, m12, m23, m13)

    ### Inverse should be [1/2, 1/3, 1/4] on diagonal
    assert np.isclose(a11, 0.5)
    assert np.isclose(a22, 1 / 3)
    assert np.isclose(a33, 0.25)
    assert np.isclose(a12, 0.0)
    assert np.isclose(a23, 0.0)
    assert np.isclose(a13, 0.0)


def test_inv_symmetric_3x3_general():
    """Test inverse of general symmetric 3x3 matrix."""
    ### Create a symmetric matrix
    m11, m22, m33 = 4.0, 5.0, 6.0
    m12, m23, m13 = 1.0, 2.0, 1.5

    a11, a22, a33, a12, a23, a13 = inv_symmetric_3x3(m11, m22, m33, m12, m23, m13)

    ### Construct matrices
    M = np.array([[m11, m12, m13], [m12, m22, m23], [m13, m23, m33]])
    A = np.array([[a11, a12, a13], [a12, a22, a23], [a13, a23, a33]])

    ### M @ A should be identity
    product = M @ A
    assert np.allclose(product, np.eye(3), rtol=1e-6)


def test_inv_symmetric_3x3_against_numpy():
    """Test that inv_symmetric_3x3 matches NumPy's inv for symmetric matrices."""
    m11, m22, m33 = 7.0, 8.0, 9.0
    m12, m23, m13 = 2.5, 1.5, 3.0

    a11, a22, a33, a12, a23, a13 = inv_symmetric_3x3(m11, m22, m33, m12, m23, m13)

    M = np.array([[m11, m12, m13], [m12, m22, m23], [m13, m23, m33]])
    M_inv_numpy = np.linalg.inv(M)

    A = np.array([[a11, a12, a13], [a12, a22, a23], [a13, a23, a33]])

    assert np.allclose(A, M_inv_numpy, rtol=1e-6)


def test_norm_inf_order():
    """Test infinity norm."""
    x = np.array([3, -7, 2], dtype=float)

    result = norm(x, ord=np.inf)
    expected = 7.0

    assert np.isclose(result, expected)


def test_outer_mixed_types():
    """Test outer product with mixed NumPy and CasADi types."""
    x = np.array([1, 2, 3])
    y = cas.DM([4, 5])

    result = outer(x, y)
    expected = np.array([[4, 5], [8, 10], [12, 15]])

    assert np.allclose(np.array(result), expected, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
