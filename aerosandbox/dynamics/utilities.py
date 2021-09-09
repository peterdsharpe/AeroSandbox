def inv_symmetric_3x3(
        m11,
        m22,
        m33,
        m12,
        m23,
        m13,
):
    """
    Explicitly computes the inverse of a symmetric 3x3 matrix.

    Input matrix (note symmetry):
    
    [m11, m12, m13]
    [m12, m22, m23]
    [m13, m23, m33]
    
    Output matrix (note symmetry):
    
    [a11, a12, a13]
    [a12, a22, a23]
    [a13, a23, a33]

    From https://math.stackexchange.com/questions/233378/inverse-of-a-3-x-3-covariance-matrix-or-any-positive-definite-pd-matrix
    """
    inv_det = 1 / (
            m11 * (m33 * m22 - m23 ** 2) -
            m12 * (m33 * m12 - m23 * m13) +
            m13 * (m23 * m12 - m22 * m13)
    )
    a11 = m33 * m22 - m23 ** 2
    a12 = m13 * m23 - m33 * m12
    a13 = m12 * m23 - m13 * m22

    a22 = m33 * m11 - m13 ** 2
    a23 = m12 * m13 - m11 * m23

    a33 = m11 * m22 - m12 ** 2

    a11 = a11 * inv_det
    a12 = a12 * inv_det
    a13 = a13 * inv_det
    a22 = a22 * inv_det
    a23 = a23 * inv_det
    a33 = a33 * inv_det

    return a11, a22, a33, a12, a23, a13
