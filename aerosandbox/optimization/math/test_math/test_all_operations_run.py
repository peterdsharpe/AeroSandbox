from aerosandbox import Opti
import pytest
from aerosandbox.optimization.math import *
import numpy as np

### Cause warnings to raise exceptions, to make this bulletproof
np.seterr(all='raise')


@pytest.fixture
def types():
    ### NumPy data types
    scalar_np = array(1)
    vector_np = array([1, 1])
    matrix_np = np.ones((2, 2))

    ### CasADi data types
    opti = Opti()
    scalar_cas = opti.variable(init_guess=1)
    vector_cas = opti.variable(n_vars=2, init_guess=1)

    ### Dynamically-typed data type creation (i.e. type depends on inputs)
    vector_dynamic = array([scalar_cas, scalar_cas])  # vector as a dynamic-typed array
    matrix_dynamic = array([  # matrix as an dynamic-typed array
        [scalar_cas, scalar_cas],
        [scalar_cas, scalar_cas]
    ])

    ### Create lists of possible variable types for scalars, vectors, and matrices.
    scalar_options = [scalar_cas, scalar_np]
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
    for option_set in types.values():
        for x in option_set:
            for y in option_set:
                ### Comparisons
                x == y
                x != y
                x > y
                x >= y
                x < y
                x <= y

                ### Conditionals
                if_else(
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
        clip(x, 0, 1)


def test_vector_math(types):
    pass


def test_spacing(types):
    for x in types["scalar"]:
        linspace(x - 1, x + 1, 10)
        cosspace(x - 1, x + 1, 10)

        # ### Vector math
        # v.T @ v  # Inner product
        # v @ v.T  # Warning: this does NOT give you an outer product
        # inner(v, v)  # Inner product
        # outer(v, v)  # Outer product
        # # np.matmul(v, v) # Note: use v @ v notation instead.
        # norm(v)
        #
        # ### Matrix math
        # # np.transpose(m)  # Note: use m.T or v.T instead.
        # m.T
        # v.T
        # m @ m
        # m @ v  # TODO investigate this
        # # v.T @ m  # TODO investigate this
        # linear_solve(m, v)  # TODO get this working


if __name__ == '__main__':
    pytest.main()
