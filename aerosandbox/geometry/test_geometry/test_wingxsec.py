from aerosandbox import *
import pytest


def test_init():  # TODO actually test this
    WingXSec(
        xyz_le=np.array([0, 0, 0]),
        chord=1.0,
        twist=0,
        airfoil=Airfoil("naca0012"),
        control_surface_is_symmetric=True,
        control_surface_hinge_point=0.75,
        control_surface_deflection=0.0,
    )


if __name__ == "__main__":
    pytest.main()
