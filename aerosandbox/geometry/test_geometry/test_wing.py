from aerosandbox import Wing, WingXSec, Airfoil, ControlSurface
import aerosandbox.numpy as np
import pytest


def w() -> Wing:
    wing = Wing(
        name="MyWing",
        xsecs=[
            WingXSec(
                xyz_le=np.array([1, 1, 0]),
                chord=0.5,
                twist=5,
                airfoil=Airfoil("mh60"),
                control_surfaces=[
                    ControlSurface(
                        symmetric=True,
                    )
                ],
            ),
            WingXSec(
                xyz_le=np.array([2, 2, 0]),
                chord=0.5,
                twist=5,
                airfoil=Airfoil("mh60"),
                control_surfaces=[ControlSurface(symmetric=True)],
            ),
        ],
        symmetric=True,
    ).translate(np.array([1, 2, 3]))
    return wing


def test_span():
    assert w().span() == pytest.approx(2)


def test_area():
    assert w().area() == pytest.approx(1)


def test_aspect_ratio():
    assert w().aspect_ratio() == pytest.approx(4)


def test_is_entirely_symmetric():
    assert w().is_entirely_symmetric()


def test_mean_geometric_chord():
    assert w().mean_geometric_chord() == pytest.approx(0.5)


def test_mean_aerodynamic_chord():
    assert w().mean_aerodynamic_chord() == pytest.approx(0.5)


def test_mean_twist_angle():
    assert w().mean_twist_angle() == pytest.approx(5)


def test_mean_sweep_angle():
    assert w().mean_sweep_angle() == pytest.approx(45)


def test_aerodynamic_center():
    ac = w().aerodynamic_center()

    assert ac[0] == pytest.approx(1 + 1.5 + 1 / 8, abs=2e-2)
    assert ac[1] == pytest.approx(0)
    assert ac[2] == pytest.approx(3, abs=2e-2)


def test_mesh_line_with_iterable_x_nondim():
    """
    mesh_line() should accept an iterable x_nondim (one value per xsec) with
    add_camber=True; each xsec's camber must be evaluated at that xsec's own
    x_nondim value. Previously, this raised a broadcast ValueError (or gave
    wrong points) because the full x_nondim iterable was passed to
    local_camber().
    """
    wing = Wing(
        xsecs=[
            WingXSec(xyz_le=[0, 0, 0], chord=1.0, airfoil=Airfoil("naca4412")),
            WingXSec(xyz_le=[0, 1, 0], chord=0.8, airfoil=Airfoil("naca4412")),
            WingXSec(xyz_le=[0.1, 2, 0], chord=0.5, airfoil=Airfoil("naca4412")),
            WingXSec(xyz_le=[0.2, 3, 0], chord=0.3, airfoil=Airfoil("naca4412")),
        ]
    )

    x_nondims = [0.25, 0.3, 0.35, 0.4]
    points = wing.mesh_line(x_nondim=x_nondims, add_camber=True)

    assert len(points) == len(wing.xsecs)

    ### Each point should match the scalar-input result at that cross-section:
    for i, x_nondim in enumerate(x_nondims):
        points_scalar = wing.mesh_line(x_nondim=x_nondim, add_camber=True)
        assert np.allclose(points[i], points_scalar[i])

    ### Wrong-length iterables should be rejected:
    with pytest.raises(ValueError):
        wing.mesh_line(x_nondim=[0.25, 0.3], add_camber=True)


if __name__ == "__main__":
    pytest.main()
