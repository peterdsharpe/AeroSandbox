import aerosandbox as asb
import aerosandbox.numpy as np
import pytest

pv = pytest.importorskip("pyvista")
pv.OFF_SCREEN = True  # Ensure no GUI window is needed (headless-safe)


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


if __name__ == "__main__":
    pytest.main([__file__])
