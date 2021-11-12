from aerosandbox.dynamics.point_mass.common_point_mass import _DynamicsPointMassBaseClass
import aerosandbox.numpy as np
from typing import Union


class DynamicsPointMass2DCartesian(_DynamicsPointMassBaseClass):
    """
    Dynamics instance:
    * simulating a point mass
    * in 2D
    * with velocity parameterized in cartesian coordinates

    State variables:
        x_e: x-position, in Earth axes. [meters]
        z_e: z-position, in Earth axes. [meters]
        u_e: x-velocity, in Earth axes. [m/s]
        w_e: z-velocity, in Earth axes. [m/s]

    Control variables:
        Fx_e: Force along the Earth-x axis. [N]
        Fz_e: Force along the Earth-z axis. [N]

    """

    def __init__(self,
                 x_e: Union[np.ndarray, float],
                 z_e: Union[np.ndarray, float],
                 u_e: Union[np.ndarray, float],
                 w_e: Union[np.ndarray, float],
                 ):
        self.x_e = x_e
        self.z_e = z_e
        self.u_e = u_e
        self.w_e = w_e


if __name__ == '__main__':
    dyn = DynamicsPointMass2DCartesian()
