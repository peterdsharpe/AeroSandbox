from aerosandbox.dynamics.point_mass.point_3D.cartesian import DynamicsPointMass3DCartesian
from aerosandbox.weights.mass_properties import MassProperties
import aerosandbox.numpy as np
from typing import Union, Dict, Tuple


class DynamicsPointMass1DHorizontal(DynamicsPointMass3DCartesian):
    """
    Dynamics instance:
    * simulating a point mass
    * in 1D, oriented horizontally (i.e., the .add_gravity() method will have no effect)

    State variables:
        x_e: x-position, in Earth axes. [meters]
        u_e: x-velocity, in Earth axes. [m/s]

    Control variables:
        Fx_e: Force along the Earth-x axis. [N]

    """

    def __init__(self,
                 mass_props: MassProperties = None,
                 x_e: Union[np.ndarray, float] = 0,
                 u_e: Union[np.ndarray, float] = 0,
                 ):
        # Initialize state variables
        self.mass_props = MassProperties() if mass_props is None else mass_props
        self.x_e = x_e
        self.y_e = 0
        self.z_e = 0
        self.u_e = u_e
        self.v_e = 0
        self.w_e = 0

        # Initialize indirect control variables
        self.alpha = 0
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
            "u_e": self.u_e,
        }

    @property
    def control_variables(self) -> Dict[str, Union[float, np.ndarray]]:
        return {
            "Fx_e": self.Fx_e,
        }

    def state_derivatives(self) -> Dict[str, Union[float, np.ndarray]]:
        derivatives = super().state_derivatives()
        return {
            k: derivatives[k] for k in self.state.keys()
        }


if __name__ == '__main__':
    dyn = DynamicsPointMass1DHorizontal()
