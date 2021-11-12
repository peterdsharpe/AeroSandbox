from aerosandbox.dynamics.point_mass.common_point_mass import _DynamicsPointMassBaseClass
import aerosandbox.numpy as np
from typing import Union


class DynamicsPointMass2DSpeedGamma(_DynamicsPointMassBaseClass):
    def __init__(self,
                 x_e: Union[np.ndarray, float],
                 z_e: Union[np.ndarray, float],
                 V: Union[np.ndarray, float],
                 gamma: Union[np.ndarray, float],
                 ):
        self.x_e = x_e
        self.z_e = z_e
        self.V = V
        self.gamma = gamma


if __name__ == '__main__':
    dyn = DynamicsPointMass2DSpeedGamma()
