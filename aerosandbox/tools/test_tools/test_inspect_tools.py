from aerosandbox.tools.inspect_tools import *
import pytest


def test_function_argument_names_from_source_code():
    tests = {  # Pairs of {input: expected_output}
        "f(a, b)"              : ['a', 'b'],
        "f(a,b)"               : ['a', 'b'],
        "f(\na,\nb)"           : ['a', 'b'],
        "g = f(a, b)"          : ['a', 'b'],
        "g.h = f(a, b)"        : ['a', 'b'],
        "g.h() = f(a, b)"      : ['a', 'b'],
        "g.h(i=j) = f(a, b)"   : ['a', 'b'],
        "f(a, b) + g(h)"       : ['a', 'b'],
        "f(a: int, b: float())": ['a', 'b'],
        "f(a, b).g(c, d)"      : ['a', 'b'],
        "f(a(b), c)"           : ['a(b)', 'c'],
        "f(a(b,c), d)"         : ['a(b,c)', 'd'],
        "f({a:b}, c)"          : ['{a:b}', 'c'],
        "f(a[b], c)"           : ['a[b]', 'c'],
        "f({a:b, c:d}, e)"     : ['{a:b,c:d}', 'e'],
        "f({a:b,\nc:d}, e)"    : ['{a:b,c:d}', 'e'],
        "f(dict(a=b,c=d), e)"  : ['dict(a=b,c=d)', 'e'],
        "f(a=1, b=2)"          : ['a=1', 'b=2'],
    }
    for input, expected_output in tests.items():
        assert get_function_argument_names_from_source_code(input) == expected_output

    with pytest.raises(ValueError):
        get_function_argument_names_from_source_code(
            "3 + 5"
        )
    with pytest.raises(ValueError):
        get_function_argument_names_from_source_code(
            ""
        )


if __name__ == '__main__':
    pytest.main()
