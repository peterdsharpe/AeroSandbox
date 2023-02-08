from aerosandbox.dynamics.point_mass.point_3D.cartesian import DynamicsPointMass3DCartesian
from aerosandbox.weights.mass_properties import MassProperties
import aerosandbox.numpy as np
from typing import Union, Dict, Tuple


class DynamicsPointMass2DCartesian(DynamicsPointMass3DCartesian):
    """
    Dynamics instance:
    * simulating a point mass
    * in 2D
    * with velocity parameterized in Cartesian coordinates

    State variables:
        x_e: x-position, in Earth axes. [meters]
        z_e: z-position, in Earth axes. [meters]
        u_e: x-velocity, in Earth axes. [m/s]
        w_e: z-velocity, in Earth axes. [m/s]

    Indirect control variables:
        alpha: Angle of attack. [degrees]

    Control variables:
        Fx_e: Force along the Earth-x axis. [N]
        Fz_e: Force along the Earth-z axis. [N]

    """

    def __init__(self,
                 mass_props: MassProperties = None,
                 x_e: Union[float, np.ndarray] = 0,
                 z_e: Union[float, np.ndarray] = 0,
                 u_e: Union[float, np.ndarray] = 0,
                 w_e: Union[float, np.ndarray] = 0,
                 alpha: Union[float, np.ndarray] = 0,
                 ):
        # Initialize state variables
        self.mass_props = MassProperties() if mass_props is None else mass_props
        self.x_e = x_e
        self.y_e = 0
        self.z_e = z_e
        self.u_e = u_e
        self.v_e = 0
        self.w_e = w_e

        # Initialize indirect control variables
        self.alpha = alpha
        self.beta = 0
        self.bank = 0

        # Initialize control variables
        self.Fx_e = 0
        self.Fy_e = 0
        self.Fz_e = 0

    @property
    def state(self) -> Dict[str, Union[float, np.ndarray]]:
        return {
            "x_e": self.x_e,
            "z_e": self.z_e,
            "u_e": self.u_e,
            "w_e": self.w_e,
        }

    @property
    def control_variables(self) -> Dict[str, Union[float, np.ndarray]]:
        return {
            "alpha": self.alpha,
            "Fx_e" : self.Fx_e,
            "Fz_e" : self.Fz_e,
        }

    def state_derivatives(self) -> Dict[str, Union[float, np.ndarray]]:
        derivatives = super().state_derivatives()
        return {
            k: derivatives[k] for k in self.state.keys()
        }


if __name__ == '__main__':
    dyn = DynamicsPointMass2DCartesian()
