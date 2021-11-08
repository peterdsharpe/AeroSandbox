from aerosandbox.dynamics.common import _DynamicsBaseClass
import aerosandbox.numpy as np
from typing import Union

class DynamicsPointMass2D(_DynamicsBaseClass):
    def __init__(self,
                 xe: Union[np.ndarray, float],
                 ze: Union[np.ndarray, float],
                 V: Union[np.ndarray, float],
                 gamma: Union[np.ndarray, float],
                 ):
        self.xe = xe
        self.ze = ze
        self.V = V
        self.gamma = gamma

if __name__ == '__main__':
    d = DynamicsPointMass2D()