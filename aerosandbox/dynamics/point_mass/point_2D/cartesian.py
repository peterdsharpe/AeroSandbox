from aerosandbox.dynamics.point_mass.common_point_mass import _DynamicsPointMassBaseClass
from aerosandbox.weights.mass_properties import MassProperties
import aerosandbox.numpy as np
from typing import Union, Dict, Tuple


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
                 mass_props: MassProperties = None,
                 x_e: Union[np.ndarray, float] = 0,
                 z_e: Union[np.ndarray, float] = 0,
                 u_e: Union[np.ndarray, float] = 0,
                 w_e: Union[np.ndarray, float] = 0,
                 ):
        # Initialize state variables
        self.mass_props = MassProperties() if mass_props is None else mass_props
        self.x_e = x_e
        self.z_e = z_e
        self.u_e = u_e
        self.w_e = w_e

        # Initialize control variables
        self.Fx_e = 0
        self.Fz_e = 0

    def state(self) -> Dict[str, Union[float, np.ndarray]]:
        pass

    def control_variables(self) -> Dict[str, Union[float, np.ndarray]]:
        pass

    def state_derivatives(self) -> Dict[str, Union[float, np.ndarray]]:
        pass

    def convert_axes(self,
                     x_from: float,
                     y_from: float,
                     z_from: float,
                     from_axes: str,
                     to_axes: str,
                     ) -> Tuple[float, float, float]:
        pass

    def add_force(self,
                  Fx: Union[np.ndarray, float] = 0,
                  Fy: Union[np.ndarray, float] = 0,
                  Fz: Union[np.ndarray, float] = 0,
                  axes="wind",
                  ) -> None:
        pass

    @property
    def speed(self) -> float:
        return (
            self.u_e ** 2 +
            self.z_e ** 2
        ) ** 0.5

if __name__ == '__main__':
    dyn = DynamicsPointMass2DCartesian()
