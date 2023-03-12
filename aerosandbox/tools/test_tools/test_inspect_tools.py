from aerosandbox.tools.inspect_tools import *
import pytest
import inspect


def test_function_argument_names_from_source_code():
    tests = {  # Pairs of {input: expected_output}
        "f(a, b)"               : ['a', 'b'],
        "f(a,b)"                : ['a', 'b'],
        "f(\na,\nb)"            : ['a', 'b'],
        "g = f(a, b)"           : ['a', 'b'],
        "g.h = f(a, b)"         : ['a', 'b'],
        "g.h() = f(a, b)"       : ['a', 'b'],
        "g.h(i=j) = f(a, b)"    : ['a', 'b'],
        "f(a, b) + g(h)"        : ['a', 'b'],
        "f(a: int, b: MyType())": ['a', 'b'],
        "f(a, b).g(c, d)"       : ['a', 'b'],
        "f(a(b), c)"            : ['a(b)', 'c'],
        "f(a(b,c), d)"          : ['a(b,c)', 'd'],
        "f({a:b}, c)"           : ['{a:b}', 'c'],
        "f(a[b], c)"            : ['a[b]', 'c'],
        "f({a:b, c:d}, e)"      : ['{a:b, c:d}', 'e'],
        "f({a:b,\nc:d}, e)"     : ['{a:b,c:d}', 'e'],
        "f(dict(a=b,c=d), e)"   : ['dict(a=b,c=d)', 'e'],
        "f(a=1, b=2)"           : ['a=1', 'b=2'],
        "f()"                   : [''],
        "f(a, [i for i in l])"  : ['a', '[i for i in l]'],
        "f(incomplete, "        : ValueError,
        "3 + 5"                 : ValueError,
        ""                      : ValueError,
    }

    for input, expected_output in tests.items():

        ### If you're expecting an error, make sure it gets raised
        if inspect.isclass(expected_output) and issubclass(expected_output, Exception):
            with pytest.raises(expected_output):
                get_function_argument_names_from_source_code(input)

        ### If you're expecting a specific output, make sure you get that
        else:
            assert get_function_argument_names_from_source_code(input) == expected_output


def test_codegen_builtins():
    for x in [
        1,
        1.0,
        "hello",
        True,
        None,
        [1, 2, 3],
        (1, 2, 3),
        {1, 2, 3},
        {1: 2, 3: 4},
        dict(cat=1, dog=2),
        [1, 2, [3, 4, [5, 6]]],
    ]:

        code, imports = codegen(x)
        for import_str in imports:
            exec(import_str)
        assert eval(code) == x


def test_codegen_numpy():
    import numpy as _np  # doing this to hide the import from the codegen function

    for x in [
        _np.array([1, 2, 3]),
        _np.arange(10),
        _np.arange(12).reshape(3, 4),
    ]:

        code, imports = codegen(x)
        for import_str in imports:
            exec(import_str)
        assert (eval(code) == x).all()


if __name__ == '__main__':
    test_function_argument_names_from_source_code()
    pytest.main()
