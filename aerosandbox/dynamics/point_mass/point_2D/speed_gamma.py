from aerosandbox.dynamics.point_mass.common_point_mass import _DynamicsPointMassBaseClass
from aerosandbox.weights.mass_properties import MassProperties
import aerosandbox.numpy as np
from typing import Union


class DynamicsPointMass2DSpeedGamma(_DynamicsPointMassBaseClass):
    """
    Dynamics instance:
    * simulating a point mass
    * in 2D
    * with velocity parameterized in speed-gamma space.

    State variables:
        x_e: x-position, in Earth axes. [meters]
        z_e: z-position, in Earth axes. [meters]
        speed: Speed; equivalent to u_w, the x-velocity in wind axes. [m/s]
        gamma: Flight path angle. [rad]

    Control variables:
        Fx_w: Force along the wind-x axis. [N]
        Fz_w: Force along the wind-z axis. [N]
    """

    def __init__(self,
                 mass_props: MassProperties = None,
                 x_e: Union[np.ndarray, float] = 0,
                 z_e: Union[np.ndarray, float] = 0,
                 speed: Union[np.ndarray, float] = 0,
                 gamma: Union[np.ndarray, float] = 0,
                 ):
        # Initialize state variables
        self.mass_props = MassProperties() if mass_props is None else mass_props
        self.x_e = x_e
        self.z_e = z_e
        self.speed = speed
        self.gamma = gamma

        # Initialize control variables
        self.Fx_w = 0
        self.Fz_w = 0

    @property
    def speed(self) -> float:
        return self.speed


if __name__ == '__main__':
    dyn = DynamicsPointMass2DSpeedGamma()
