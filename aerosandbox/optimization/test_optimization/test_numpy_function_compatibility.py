from aerosandbox import *
import pytest


def test_numpy_function_compatibility():
    ### CasADi data types
    opti = Opti()
    scalar_cas = opti.variable()
    vector_cas = opti.variable(n_vars=2)

    ### NumPy data types
    scalar_np = np.array(1)
    vector_np = np.array([1, 1])
    matrix_np = np.eye(2)

    ### Mixed data type creation
    vector_mixed = np.array([scalar_cas, scalar_cas])  # vector as an object array
    matrix_mixed = np.array([  # matrix as an object array
        [scalar_cas, scalar_cas],
        [scalar_cas, scalar_cas]
    ])

    for s in [scalar_cas, scalar_np]:
        for v in [vector_cas, vector_np, vector_mixed]:
            for m in [matrix_np, matrix_mixed]:

                ### Indexing
                v[0] + v[1]
                v[1:] + v[:-1]

                ### Stacking
                # np.hstack((x, y))  # Can't do this raw
                # np.hstack((np.array([scalar]), np.array([scalar])))
                # np.hstack((v, v))
                # np.concatenate((v, v))

                ### Test math operators (https://numpy.org/doc/stable/reference/ufuncs.html)
                v + v
                v - v
                v * v
                v / v
                v ** v
                np.exp(v)
                np.log(v)
                np.log10(v)
                np.fabs(v)  # Note: do this instead of np.abs()
                np.sqrt(v)  # Note: do x ** 0.5 rather than np.sqrt(x).

                ### Test trig
                np.sin(v)
                np.cos(v)
                np.tan(v)
                np.arcsin(v)
                np.arccos(v)
                np.arctan(v)
                np.arctan2(s, s)  # Note: does not work with (v, v) arguments.
                np.sinh(v)
                np.cosh(v)
                np.tanh(v)
                np.arcsinh(v)
                np.arccosh(v)
                np.arctanh(v - 0.5)  # `- 0.5` to give valid argument

                ### Logical operators do not work due to non-numeric values
                # a >= a
                # a > a
                # a <= a
                # a < a
                # a == a
                # a != a

                ### Floating functions
                np.fabs(v)

                ### Vector math
                v.T @ v  # Inner product
                v @ v.T  # Outer product
                # np.inner(v, v) # Note: do v.T @ v instead
                # np.outer(v, v) # Note: do v @ v.T instead
                # np.matmul(v, v) # Note: use v @ v notation instead.
                np.linalg.norm(np.array([v]))

                ### Matrix math
                # np.transpose(m)  # Note: use m.T or v.T instead.
                m.T
                v.T
                m @ m
                # m @ v # TODO investigate this
                # v.T @ m # TODO investigate this
                # np.einsum("ij,j", m, v)  # Use `m @ v` notation instead
                # np.linalg.solve

                ### Utilities
                cas.linspace(scalar_cas - 1, scalar_cas + 1, 10)


if __name__ == '__main__':
    pytest.main()
