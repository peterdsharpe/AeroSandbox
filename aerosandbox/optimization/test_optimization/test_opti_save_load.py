import aerosandbox as asb
import aerosandbox.numpy as np
import pytest
import os

"""
Minimization over a simple unimodal function is used here:
    Minimize (x-1) ** 2 + (y-1) ** 2
"""


def sumsqr(x):
    return np.sum(x ** 2)


def test_opti():
    opti = asb.Opti()  # set up an optimization environment

    # Define optimization variables
    x = opti.variable(init_guess=0)
    y = opti.variable(init_guess=0)

    # Define objective
    f = (x - 1) ** 2 + (y - 1) ** 2
    opti.minimize(f)

    # Optimize
    sol = opti.solve()

    for i in [x, y]:
        assert sol(i) == pytest.approx(1, abs=1e-4)


def test_save_opti(tmp_path):
    temp_filename = tmp_path / "temp.json"

    opti = asb.Opti(cache_filename=temp_filename)  # set up an optimization environment

    # Define optimization variables
    x = opti.variable(init_guess=0)
    y = opti.variable(init_guess=0)

    # Define objective
    f = (x - 1) ** 2 + (y - 1) ** 2
    opti.minimize(f)

    # Optimize
    sol = opti.solve()

    opti.save_solution()


def test_save_and_load_opti(tmp_path):
    temp_filename = tmp_path / "temp.json"

    ### Round 1 optimization: free optimization

    opti = asb.Opti(
        cache_filename=temp_filename,
        save_to_cache_on_solve=True,
    )
    x = opti.variable(init_guess=0, category="Cat 1")
    y = opti.variable(init_guess=0, category="Cat 2")
    f = (x - 1) ** 2 + (y - 1) ** 2
    opti.minimize(f)

    # Optimize, save to cache, print
    sol = opti.solve()
    opti.save_solution()
    for i in ["x", "y", "f"]:
        print(f"{i}: {sol(eval(i))}")

    # Test
    assert sol(x) == pytest.approx(1)
    assert sol(y) == pytest.approx(1)
    assert sol(f) == pytest.approx(0)

    ### Round 2 optimization: Cat 1 is fixed from before; slightly different objective now
    opti = asb.Opti(
        cache_filename=temp_filename,
        variable_categories_to_freeze=["Cat 1"],
        load_frozen_variables_from_cache=True,

    )
    x = opti.variable(init_guess=0, category="Cat 1")
    y = opti.variable(init_guess=0, category="Cat 2")
    f = (x - 2) ** 2 + (y - 2) ** 2
    opti.minimize(f)

    # Optimize, save to cache, print
    sol = opti.solve()
    for i in ["x", "y", "f"]:
        print(f"{i}: {sol(eval(i))}")

    # Test
    assert sol(x) == pytest.approx(1)
    assert sol(y) == pytest.approx(2)
    assert sol(f) == pytest.approx(1)


def test_save_and_load_opti_uncategorized(tmp_path):
    temp_filename = tmp_path / "temp.json"

    ### Round 1 optimization: free optimization

    opti = asb.Opti(
        cache_filename=temp_filename,
        save_to_cache_on_solve=True,
    )
    x = opti.variable(init_guess=0)
    y = opti.variable(init_guess=0)
    f = (x - 1) ** 2 + (y - 1) ** 2
    opti.minimize(f)

    # Optimize, save to cache, print
    sol = opti.solve()
    opti.save_solution()
    for i in ["x", "y", "f"]:
        print(f"{i}: {sol(eval(i))}")

    # Test
    assert sol(x) == pytest.approx(1)
    assert sol(y) == pytest.approx(1)
    assert sol(f) == pytest.approx(0)

    ### Round 2 optimization: Cat 1 is fixed from before; slightly different objective now
    opti = asb.Opti(
        cache_filename=temp_filename,
        variable_categories_to_freeze=["Uncategorized"],
        load_frozen_variables_from_cache=True,

    )
    x = opti.variable(init_guess=0)
    y = opti.variable(init_guess=0)
    f = (x - 2) ** 2 + (y - 2) ** 2
    opti.minimize(f)

    # Optimize, save to cache, print
    sol = opti.solve()
    for i in ["x", "y", "f"]:
        print(f"{i}: {sol(eval(i))}")

    # Test
    assert sol(x) == pytest.approx(1)
    assert sol(y) == pytest.approx(1)
    assert sol(f) == pytest.approx(2)


def test_save_and_load_opti_vectorized(tmp_path):
    temp_filename = tmp_path / "temp.json"

    ### Round 1 optimization: free optimization

    opti = asb.Opti(
        cache_filename=temp_filename,
        variable_categories_to_freeze=[],
        save_to_cache_on_solve=True,
    )
    x = opti.variable(init_guess=0, n_vars=3, category="Cat 1")
    y = opti.variable(init_guess=0, n_vars=3, category="Cat 2")
    f = sumsqr(x - 1) + sumsqr(y - 2)
    opti.minimize(f)

    # Optimize, save to cache, print
    sol = opti.solve()
    opti.save_solution()
    for i in ["x", "y", "f"]:
        print(f"{i}: {sol(eval(i))}")

    # Test
    assert sol(x) == pytest.approx(1)
    assert sol(y) == pytest.approx(2)
    assert sol(f) == pytest.approx(0)

    ### Round 2 optimization: Cat 1 is fixed from before; slightly different objective now
    opti = asb.Opti(
        cache_filename=temp_filename,
        variable_categories_to_freeze=["Cat 1"],
        load_frozen_variables_from_cache=True,

    )
    x = opti.variable(init_guess=0, n_vars=3, category="Cat 1")
    y = opti.variable(init_guess=0, n_vars=3, category="Cat 2")
    f = sumsqr(x - 3) + sumsqr(y - 4)
    opti.minimize(f)

    # Optimize, save to cache, print
    sol = opti.solve()
    for i in ["x", "y", "f"]:
        print(f"{i}: {sol(eval(i))}")

    # Test
    assert sol(x) == pytest.approx(1)
    assert sol(y) == pytest.approx(4)
    assert sol(f) == pytest.approx(12)


def test_save_and_load_opti_freeze_override(tmp_path):
    temp_filename = tmp_path / "temp.json"

    ### Round 1 optimization: free optimization

    opti = asb.Opti(
        cache_filename=temp_filename,
        variable_categories_to_freeze=[],
        save_to_cache_on_solve=True,
    )
    x = opti.variable(init_guess=0, category="Cat 1")
    y = opti.variable(init_guess=0, category="Cat 2")
    f = (x - 1) ** 2 + (y - 1) ** 2
    opti.minimize(f)

    # Optimize, save to cache, print
    sol = opti.solve()
    opti.save_solution()
    for i in ["x", "y", "f"]:
        print(f"{i}: {sol(eval(i))}")

    # Test
    assert sol(x) == pytest.approx(1)
    assert sol(y) == pytest.approx(1)
    assert sol(f) == pytest.approx(0)

    ### Round 2 optimization: Cat 1 is fixed from before but then overridden; slightly different objective now
    opti = asb.Opti(
        cache_filename=temp_filename,
        variable_categories_to_freeze=["Cat 1"],
        load_frozen_variables_from_cache=True,

    )
    x = opti.variable(init_guess=3, category="Cat 1", freeze=True)
    y = opti.variable(init_guess=0, category="Cat 2")
    f = (x - 2) ** 2 + (y - 2) ** 2
    opti.minimize(f)

    # Optimize, save to cache, print
    sol = opti.solve()
    for i in ["x", "y", "f"]:
        print(f"{i}: {sol(eval(i))}")

    # Test
    assert sol(x) == pytest.approx(3)
    assert sol(y) == pytest.approx(2)
    assert sol(f) == pytest.approx(1)


if __name__ == '__main__':
    # from pathlib import Path
    # tmp_path = Path.home() / "Downloads" / "test"
    # test_save_and_load_opti(tmp_path)
    pytest.main()
