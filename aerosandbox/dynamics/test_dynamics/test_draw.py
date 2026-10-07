import aerosandbox as asb
import aerosandbox.numpy as np
import pytest

pv = pytest.importorskip("pyvista")
pv.OFF_SCREEN = True  # Ensure no GUI window is needed (headless-safe)

# Off-screen VTK rendering still requires a working GL context; on a headless
# machine without one, VTK aborts the whole process (SIGABRT) rather than
# raising, so no in-test guard can catch it. Start a virtual framebuffer if
# available, and otherwise skip the module entirely.
import os
import shutil

if os.name == "posix" and not os.environ.get("DISPLAY"):
    if shutil.which("Xvfb"):
        try:
            pv.start_xvfb()
        except Exception as e:
            pytest.skip(
                f"Could not start Xvfb for pyvista rendering tests: {e}",
                allow_module_level=True,
            )
    else:
        pytest.skip(
            "No display and no Xvfb available; skipping pyvista rendering tests.",
            allow_module_level=True,
        )


def make_dyn(n_points: int) -> asb.DynamicsPointMass3DCartesian:
    return asb.DynamicsPointMass3DCartesian(
        mass_props=asb.MassProperties(mass=1),
        x_e=np.linspace(0, 100, n_points),
        y_e=np.linspace(0, 10, n_points),
        z_e=np.linspace(0, -50, n_points),
        u_e=10,
        v_e=1,
        w_e=-5,
    )


def test_draw_autoscales_vehicle_on_multipoint_trajectory():
    """
    On NumPy >= 2.0, `float()` on the 1-element array returned by `np.diff()`
    raised a TypeError when auto-computing `scale_vehicle_model`.
    """
    dyn = make_dyn(10)
    plotter = dyn.draw(show=False)
    assert plotter is not None


def test_draw_with_str_filepath_vehicle_model():
    """
    Passing `vehicle_model` as a str filepath used to discard the `pv.read()`
    result, then crash with AttributeError ('str' object has no attribute
    'bounds').
    """
    stl_path = str(
        asb._asb_root / "dynamics" / "visualization" / "default_assets" / "talon.stl"
    )
    dyn = make_dyn(10)
    plotter = dyn.draw(vehicle_model=stl_path, show=False)
    assert plotter is not None


@pytest.mark.parametrize("n_points", [1, 2, 3, 4])
def test_draw_short_trajectories(n_points):
    """
    Trajectories with 1-3 time points used to crash in draw(): the spline
    degree k was set equal to the number of points, but scipy's
    InterpolatedUnivariateSpline requires (number of points) > k.
    """
    dyn = make_dyn(n_points)
    plotter = dyn.draw(show=False)
    assert plotter is not None


def test_draw_with_invalid_str_vehicle_model_raises_valueerror():
    dyn = make_dyn(10)
    with pytest.raises(ValueError):
        dyn.draw(vehicle_model="not_a_real_file.stl", show=False)


if __name__ == "__main__":
    pytest.main([__file__])
