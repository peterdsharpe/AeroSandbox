import aerosandbox as asb
from aerosandbox import cas
import numpy as np
import pytest
import os

"""
Minimization over a simple unimodal function is used here:
    Minimize (x-1) ** 2 + (y-1) ** 2
"""
def test_opti():
    opti = asb.Opti()  # set up an optimization environment

    # Define optimization variables
    x = opti.variable()
    y = opti.variable()

    # Define objective
    f = (x-1) ** 2 + (y-1) ** 2
    opti.minimize(f)

    # Optimize
    sol = opti.solve()

    for i in [x, y]:
        assert sol.value(i) == pytest.approx(1, abs=1e-4)

def test_save_opti(tmp_path):
    temp_filename = tmp_path / "temp.json"

    opti = asb.Opti(cache_filename=temp_filename)  # set up an optimization environment

    # Define optimization variables
    x = opti.variable()
    y = opti.variable()

    # Define objective
    f = (x-1) ** 2 + (y-1) ** 2
    opti.minimize(f)

    # Optimize
    sol = opti.solve()

    opti.save_solution()

def test_save_and_load_opti(tmp_path):
    temp_filename = tmp_path / "temp.json"

    ### Round 1 optimization: free optimization

    opti = asb.Opti(
        cache_filename=temp_filename,
        variable_categories_to_freeze=[],
        save_to_cache_on_solve=True,
    )
    x = opti.variable(category="Cat 1")
    y = opti.variable(category="Cat 2")
    f = (x-1) ** 2 + (y-1) ** 2
    opti.minimize(f)

    # Optimize, save to cache, print
    sol = opti.solve()
    opti.save_solution()
    for i in ["x", "y", "f"]:
        print(f"{i}: {sol.value(eval(i))}")

    # Test
    assert sol.value(x) == pytest.approx(1)
    assert sol.value(y) == pytest.approx(1)
    assert sol.value(f) == pytest.approx(0)

    ### Round 2 optimization: Cat 1 is fixed from before; slightly different objective now
    opti = asb.Opti(
        cache_filename=temp_filename,
        variable_categories_to_freeze=["Cat 1"],
        load_frozen_variables_from_cache=True,

    )
    x = opti.variable(category="Cat 1")
    y = opti.variable(category="Cat 2")
    f = (x-2) ** 2 + (y-2) ** 2
    opti.minimize(f)

    # Optimize, save to cache, print
    sol = opti.solve()
    for i in ["x", "y", "f"]:
        print(f"{i}: {sol.value(eval(i))}")

    # Test
    assert sol.value(x) == pytest.approx(1)
    assert sol.value(y) == pytest.approx(2)
    assert sol.value(f) == pytest.approx(1)

def test_save_and_load_opti_vectorized(tmp_path):
    temp_filename = tmp_path / "temp.json"

    ### Round 1 optimization: free optimization

    opti = asb.Opti(
        cache_filename=temp_filename,
        variable_categories_to_freeze=[],
        save_to_cache_on_solve=True,
    )
    x = opti.variable(n_vars=3, category="Cat 1")
    y = opti.variable(n_vars=3, category="Cat 2")
    f = cas.sumsqr(x-1) + cas.sumsqr(y-2)
    opti.minimize(f)

    # Optimize, save to cache, print
    sol = opti.solve()
    opti.save_solution()
    for i in ["x", "y", "f"]:
        print(f"{i}: {sol.value(eval(i))}")

    # Test
    assert sol.value(x) == pytest.approx(1)
    assert sol.value(y) == pytest.approx(2)
    assert sol.value(f) == pytest.approx(0)

    ### Round 2 optimization: Cat 1 is fixed from before; slightly different objective now
    opti = asb.Opti(
        cache_filename=temp_filename,
        variable_categories_to_freeze=["Cat 1"],
        load_frozen_variables_from_cache=True,

    )
    x = opti.variable(n_vars=3, category="Cat 1")
    y = opti.variable(n_vars=3, category="Cat 2")
    f = cas.sumsqr(x-3) + cas.sumsqr(y-4)
    opti.minimize(f)

    # Optimize, save to cache, print
    sol = opti.solve()
    for i in ["x", "y", "f"]:
        print(f"{i}: {sol.value(eval(i))}")

    # Test
    assert sol.value(x) == pytest.approx(1)
    assert sol.value(y) == pytest.approx(4)
    assert sol.value(f) == pytest.approx(12)


if __name__ == '__main__':
    pytest.main()
    # test_save_and_load_opti_vectorized()