from aerosandbox import *
import pytest

def test_init(): # TODO actually test this
    xsec = WingXSec(
                 xyz_le = np.array([0, 0, 0]),
                 chord = 1.,
                 twist_angle = 0,
                 twist_axis = np.array([0, 1, 0]),
                 airfoil = Airfoil("naca0012"),
                 control_surface_is_symmetric = True,
                 control_surface_hinge_point = 0.75,
                 control_surface_deflection = 0.,
    )

if __name__ == '__main__':
    pytest.main()