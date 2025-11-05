import tempfile
import pytest
from aerosandbox import *


def w(name, x0, twist) -> Wing:
    return Wing(
        name=name,
        xsecs=[
            WingXSec(
                xyz_le=np.array([0, 0, 0]),
                chord=0.5,
                twist=twist - 5,
                airfoil=Airfoil("mh60"),
                control_surfaces=[ControlSurface(symmetric=True)],
            ),
            WingXSec(
                xyz_le=np.array([0, 2, 0]),
                chord=0.4,
                twist=twist,
                airfoil=Airfoil("mh60"),
                control_surfaces=[ControlSurface(symmetric=True)],
            ),
        ],
        symmetric=True,
    ).translate(np.array([x0, 0, 0]))


def f(name, x0) -> Fuselage:
    return Fuselage(
        name=name,
        xsecs=[
            FuselageXSec(xyz_c=[x0 + x, 0, 0], radius=r)
            for x, r in [(0, 0.1), (1, 0.4), (2, 0.4), (3, 0.2), (4, 0.1)]
        ],
    )


def test_cadquery_export():
    with tempfile.TemporaryDirectory() as tmpdirname:
        fname = f"{tmpdirname}/test.step"
        airplane = Airplane(
            wings=[w("FrontWing", 1, 0), w("BackWing", 3, 5)],
            fuselages=[f("Fuselage", 0)],
        )
        airplane.export_cadquery_geometry(fname, split_leading_edge=False)
        step_file = open(fname).read()
        assert "'FrontWing'" in step_file
        assert "'BackWing'" in step_file
        assert "'Fuselage'" in step_file


if __name__ == "__main__":
    a().export_cadquery_geometry("test.step")
    step_file = open("test.step").read()
