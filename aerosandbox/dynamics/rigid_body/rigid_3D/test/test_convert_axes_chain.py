import aerosandbox as asb
import aerosandbox.numpy as np
from typing import List
import copy
import pytest

vector = [1, 2, 3]
dyn = asb.DynamicsRigidBody3DBodyEuler(
    x_e=0,
    y_e=0,
    z_e=0,
    u_b=1,
    v_b=0,
    w_b=0,
    phi=0,
    theta=0,
    psi=0,
    p=0,
    q=0,
    r=0,
)


def chain_conversion(
        axes: List[str] = None
):
    if axes is None:
        axes = ["geometry", "body", "geometry"]

    x, y, z = copy.deepcopy(vector)
    for from_axes, to_axes in zip(axes, axes[1:]):
        x, y, z = dyn.convert_axes(
            x_from=x,
            y_from=y,
            z_from=z,
            from_axes=from_axes,
            to_axes=to_axes
        )
    return np.array(vector) == pytest.approx(np.array([x, y, z]))


def test_basic():
    assert chain_conversion()


def test_geometry():
    assert chain_conversion(["body", "geometry", "body"])


def test_stability():
    assert chain_conversion(["body", "stability", "body"])


def test_wind():
    assert chain_conversion(["body", "wind", "body"])


def test_earth():
    assert chain_conversion(["body", "earth", "body"])


def test_cycle():
    assert chain_conversion([
        "body",
        "geometry",
        "stability",
        "wind",
        "body",
        "wind",
        "stability",
        "geometry",
        "body",
        "geometry",
        "wind",
        "geometry",
        "stability",
        "body",
    ])


if __name__ == '__main__':
    pytest.main()
    chain_conversion()
