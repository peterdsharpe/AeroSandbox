from aerosandbox import Opti
import pytest
import aerosandbox.numpy as np

### Cause all NumPy warnings to raise exceptions, to make this bulletproof
np.seterr(all='raise')


@pytest.fixture
def types():
    ### Float data types
    scalar_float = 1.

    ### NumPy data types
    scalar_np = np.array(1)
    vector_np = np.array([1, 1])
    matrix_np = np.ones((2, 2))

    ### CasADi data types
    opti = Opti()
    scalar_cas = opti.variable(init_guess=1)
    vector_cas = opti.variable(n_vars=2, init_guess=1)

    ### Dynamically-typed data type creation (i.e. type depends on inputs)
    vector_dynamic = np.array([scalar_cas, scalar_cas])  # vector as a dynamic-typed array
    matrix_dynamic = np.array([  # matrix as an dynamic-typed array
        [scalar_cas, scalar_cas],
        [scalar_cas, scalar_cas]
    ])

    ### Create lists of possible variable types for scalars, vectors, and matrices.
    scalar_options = [scalar_float, scalar_cas, scalar_np]
    vector_options = [vector_cas, vector_np, vector_dynamic]
    matrix_options = [matrix_np, matrix_dynamic]

    return {
        "scalar": scalar_options,
        "vector": vector_options,
        "matrix": matrix_options,
        "all"   : scalar_options + vector_options + matrix_options
    }


def test_indexing_1D(types):
    for x in types["vector"] + types["matrix"]:
        x[0]  # The first element of x
        x[-1]  # The last element of x
        x[1:]  # All but the first element of x
        x[:-1]  # All but the last element of x
        x[::2]  # Every other element of x
        x[::-1]  # The elements of x, but in reversed order


def test_indexing_2D(types):
    for x in types["matrix"]:
        x[0, :]  # The first row of x
        x[:, 0]  # The first column of x
        x[1:, :]  # All but the first row of x


def test_basic_math(types):
    for x in types["all"]:
        for y in types["all"]:
            ### Arithmetic
            x + y
            x - y
            x * y
            x / y
            np.sum(x)  # Sum of all entries of array-like object x

            ### Exponentials & Powers
            x ** y
            np.power(x, y)
            np.exp(x)
            np.log(x)
            np.log10(x)
            np.sqrt(x)  # Note: do x ** 0.5 rather than np.sqrt(x).

            ### Trig
            np.sin(x)
            np.cos(x)
            np.tan(x)
            np.arcsin(x)
            np.arccos(x)
            np.arctan(x)
            np.arctan2(y, x)
            np.sinh(x)
            np.cosh(x)
            np.tanh(x)
            np.arcsinh(x)
            np.arccosh(x)
            np.arctanh(x - 0.5)  # `- 0.5` to give valid argument


def test_logic(types):
    for option_set in [
        types["scalar"],
        types["vector"],
        types["matrix"],
    ]:
        for x in option_set:
            for y in option_set:
                ### Comparisons
                """
                Note: if warnings appear here, they're from `np.array(1) == cas.MX(1)` - 
                sensitive to order, as `cas.MX(1) == np.array(1)` is fine.
                
                However, checking the outputs, these seem to be yielding correct results despite
                the warning sooo...
                """
                x == y  # Warnings coming from here
                x != y  # Warnings coming from here
                x > y
                x >= y
                x < y
                x <= y

                ### Conditionals
                np.where(
                    x > 1,
                    x ** 2,
                    0
                )

                ### Elementwise min/max
                np.fmax(x, y)
                np.fmin(x, y)

    for x in types["all"]:
        np.fabs(x)
        np.floor(x)
        np.ceil(x)
        np.clip(x, 0, 1)


def test_vector_math(types):
    for x in types["vector"]:
        for y in types["vector"]:
            x.T
            np.linalg.inner(x, y)
            np.linalg.outer(x, y)
            np.linalg.norm(x)


def test_matrix_math(types):
    for v in types["vector"]:
        for m in types["matrix"]:
            m = m + np.eye(2)  # Regularize the matrix so it's not singular

            m.T
            m @ v
            for m_other in types["matrix"]:
                m @ m_other
            np.linalg.solve(m, v)


def test_spacing(types):
    for x in types["scalar"]:
        np.linspace(x - 1, x + 1, 10)
        np.cosspace(x - 1, x + 1, 10)


def test_rotation_matrices(types):
    for angle in types["scalar"]:
        for axis in types["vector"]:
            np.rotation_matrix_2D(angle)
            np.rotation_matrix_3D(angle, np.array([axis[0], axis[1], axis[0]]))


if __name__ == '__main__':
    pytest.main()
