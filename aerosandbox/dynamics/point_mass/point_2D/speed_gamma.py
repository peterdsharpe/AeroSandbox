from aerosandbox.dynamics.point_mass.point_3D.speed_gamma_track import DynamicsPointMass3DSpeedGammaTrack
from aerosandbox.weights.mass_properties import MassProperties
import aerosandbox.numpy as np
from typing import Union, Dict, Tuple


class DynamicsPointMass2DSpeedGamma(DynamicsPointMass3DSpeedGammaTrack):
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

    Indirect control variables:
        alpha: Angle of attack. [degrees]

    Control variables:
        Fx_w: Force along the wind-x axis. [N]
        Fz_w: Force along the wind-z axis. [N]

    """

    def __init__(self,
                 mass_props: MassProperties = None,
                 x_e: Union[float, np.ndarray] = 0,
                 z_e: Union[float, np.ndarray] = 0,
                 speed: Union[float, np.ndarray] = 0,
                 gamma: Union[float, np.ndarray] = 0,
                 alpha: Union[float, np.ndarray] = 0,
                 ):
        # Initialize state variables
        self.mass_props = MassProperties() if mass_props is None else mass_props
        self.x_e = x_e
        self.y_e = 0
        self.z_e = z_e
        self.speed = speed
        self.gamma = gamma
        self.track = 0
        self.bank = 0

        # Initialize indirect control variables
        self.alpha = alpha
        self.beta = 0
        self.bank = 0

        # Initialize control variables
        self.Fx_w = 0
        self.Fy_w = 0
        self.Fz_w = 0

    @property
    def state(self) -> Dict[str, Union[float, np.ndarray]]:
        return {
            "x_e"  : self.x_e,
            "z_e"  : self.z_e,
            "speed": self.speed,
            "gamma": self.gamma,
        }

    @property
    def control_variables(self) -> Dict[str, Union[float, np.ndarray]]:
        return {
            "alpha": self.alpha,
            "Fx_w" : self.Fx_w,
            "Fz_w" : self.Fz_w,
        }

    def state_derivatives(self) -> Dict[str, Union[float, np.ndarray]]:
        derivatives = super().state_derivatives()
        return {
            k: derivatives[k] for k in self.state.keys()
        }


if __name__ == '__main__':
    dyn = DynamicsPointMass2DSpeedGamma()
