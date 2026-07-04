from aerosandbox.geometry.airfoil import Airfoil
import aerosandbox as asb
import aerosandbox.numpy as np
import pytest


@pytest.fixture
def naca4412():
    return Airfoil("naca4412")


@pytest.fixture
def e216():
    a = Airfoil("e216")
    assert a.n_points() == 61
    return a


def test_import_aerosandbox_without_plotly():
    """
    plotly is an optional dependency (only included in the `[full]` extra), so importing aerosandbox must succeed
    even when plotly is not installed. This runs in a subprocess with an import hook that simulates a missing plotly.
    """
    import os
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import sys
        import importlib.abc

        class BlockPlotly(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == "plotly" or name.startswith("plotly."):
                    raise ModuleNotFoundError(f"No module named {name!r} (simulated)")

        sys.meta_path.insert(0, BlockPlotly())

        import aerosandbox  # Should not require plotly
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=os.environ,
    )
    assert result.returncode == 0, (
        f"`import aerosandbox` failed when plotly was unavailable:\n{result.stderr}"
    )


def test_fake_airfoil():
    """
    Tests what happens when you create an airfoil that's not in the UIUC database, and you don't supply coordinates.
    """
    with pytest.warns(UserWarning):
        a = Airfoil("dae12")
    assert a.coordinates is None
    assert a.n_points() == 0


def test_coordinates_2xN_input(naca4412):
    """
    A 2xN coordinate array should be transposed into the Nx2 convention (not silently corrupted).
    """
    af = Airfoil("transposed", coordinates=naca4412.coordinates.T)
    assert np.all(af.coordinates == naca4412.coordinates)


def test_coordinates_list_input(naca4412):
    """
    A list of [x, y] pairs is a natural array-like input, and should be accepted (not silently discarded).
    """
    af = Airfoil("from_list", coordinates=naca4412.coordinates.tolist())
    assert af.coordinates is not None
    assert np.allclose(af.coordinates, naca4412.coordinates)


def test_coordinates_invalid_shape_input():
    """
    Arrays that can't be interpreted as [x, y] coordinates should raise an explicit ValueError.
    """
    with pytest.raises(ValueError):
        Airfoil("bad", coordinates=np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        Airfoil("bad", coordinates=np.ones((5, 3)))


def test_TE_angle(naca4412):
    assert naca4412.TE_angle() == pytest.approx(14.74635802332286, abs=1)


def test_local_thickness(e216):
    assert e216.local_thickness(0.5) == pytest.approx(0.08730287761717835)


def test_LE_index(e216):
    assert e216.LE_index() == 32


def test_repanel(naca4412):
    naca4412 = naca4412.repanel(n_points_per_side=300)
    assert naca4412.n_points() == 599


def test_containts_points(naca4412):
    assert naca4412.contains_points(x=0.5, y=0)
    assert np.all(
        naca4412.contains_points(x=np.array([0.5, 0.5]), y=np.array([0, -0.1]))
        == np.array([True, False])
    )
    shape = (1, 2, 3, 4)
    x_points = np.random.randn(*shape)
    y_points = np.random.randn(*shape)
    contains = naca4412.contains_points(x_points, y_points)
    assert shape == contains.shape


def test_optimize_through_control_surface_deflections():
    af = Airfoil("naca0001")

    opti = asb.Opti()

    d = opti.variable(init_guess=5, lower_bound=-90, upper_bound=90)

    afd = af.add_control_surface(deflection=d, hinge_point_x=0.75)

    opti.minimize((afd.coordinates[0, 1] - 0.2) ** 2)

    sol = opti.solve()
    # print(sol(d))
    # sol(afd).draw()

    assert sol(d) == pytest.approx(np.arcsind(-0.2 / 0.25), abs=5)


def test_optimize_through_control_surface_deflections_for_CL():
    af = Airfoil("naca0012")
    af.coordinates = af.coordinates[::5, :]

    opti = asb.Opti()

    d = opti.variable(init_guess=5, lower_bound=-90, upper_bound=90)

    aero = af.get_aero_from_neuralfoil(
        alpha=0,
        Re=1e6,
        mach=0,
        control_surfaces=[asb.ControlSurface(deflection=d, hinge_point=0.75)],
    )

    opti.minimize((aero["CL"] - 0.5) ** 2)

    sol = opti.solve()

    assert sol(d) == pytest.approx(8.34, abs=1)


if __name__ == "__main__":
    test_optimize_through_control_surface_deflections()
    # pytest.main()
