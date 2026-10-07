"""
Tests that Opti correctly tracks the source location (file + line number) where variables and
constraints are declared. This powers debugging utilities like `Opti.find_constraint_declaration()`
and `OptiSol.show_infeasibilities()`.

These tests deliberately declare things inside helper functions: if the stack-frame arithmetic is
off by one, the reported location falls back to the helper's *caller*, which these tests detect.
(At module level, an off-by-one is masked by stacklevel truncation.)
"""

import inspect
import re
from pathlib import Path

import aerosandbox as asb
import aerosandbox.numpy as np
import pytest


def get_reported_location(declaration_string: str) -> tuple[str, int]:
    """Parses '... defined in `somefile.py`, line 42:' into ('somefile.py', 42)."""
    match = re.search(r"defined in `(.+)`, line (\d+):", declaration_string)
    assert match is not None, f"Could not parse: {declaration_string}"
    return match.group(1), int(match.group(2))


THIS_FILE = Path(__file__).name


def test_constraint_declaration_single():
    opti = asb.Opti()
    x = opti.variable(init_guess=0)

    def declare(opti, x):
        lineno = inspect.currentframe().f_lineno + 1
        opti.subject_to(x >= 1)
        return lineno

    expected_lineno = declare(opti, x)

    filename, lineno = get_reported_location(
        opti.find_constraint_declaration(index=0, return_string=True)
    )
    assert filename == THIS_FILE
    assert lineno == expected_lineno


def test_constraint_declaration_list():
    ### Constraints declared via a list (the most common pattern) should report the line of the
    ### `opti.subject_to([...])` call itself - not the line of its caller. This was broken on
    ### Python 3.12+ (PEP 709 inlined the internal list comprehension, removing a stack frame).
    opti = asb.Opti()
    x = opti.variable(init_guess=0)

    def declare(opti, x):
        lineno = inspect.currentframe().f_lineno + 1
        opti.subject_to([x >= 2, x <= 10])
        return lineno

    expected_lineno = declare(opti, x)

    for index in [0, 1]:
        filename, lineno = get_reported_location(
            opti.find_constraint_declaration(index=index, return_string=True)
        )
        assert filename == THIS_FILE
        assert lineno == expected_lineno


def test_variable_declaration():
    opti = asb.Opti()

    def declare(opti):
        lineno = inspect.currentframe().f_lineno + 1
        opti.variable(init_guess=0, n_vars=3)
        return lineno

    expected_lineno = declare(opti)

    filename, lineno = get_reported_location(
        opti.find_variable_declaration(index=0, return_string=True)
    )
    assert filename == THIS_FILE
    assert lineno == expected_lineno


def test_derivative_of_variable_declaration():
    ### The derivative variable created inside `Opti.derivative_of()` should be attributed to the
    ### user's `derivative_of()` call, not to a line inside aerosandbox/optimization/opti.py.
    opti = asb.Opti()
    position = opti.variable(init_guess=0, n_vars=10)
    time = np.linspace(0, 1, 10)

    def declare(opti):
        lineno = inspect.currentframe().f_lineno + 1
        opti.derivative_of(position, with_respect_to=time, derivative_init_guess=0)
        return lineno

    expected_lineno = declare(opti)

    # The derivative variable's indices start right after `position`'s 10 entries.
    filename, lineno = get_reported_location(
        opti.find_variable_declaration(index=10, return_string=True)
    )
    assert filename == THIS_FILE
    assert lineno == expected_lineno


if __name__ == "__main__":
    pytest.main()
