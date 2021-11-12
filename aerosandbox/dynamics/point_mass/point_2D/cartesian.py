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
            "Fx_e": self.Fx_e,
            "Fz_e": self.Fz_e,
        }

    def state_derivatives(self) -> Dict[str, Union[float, np.ndarray]]:
        return {
            "x_e": self.u_e,
            "z_e": self.w_e,
            "u_e": self.Fx_e / self.mass_props.mass,
            "w_e": self.Fz_e / self.mass_props.mass,
        }

    @property
    def gamma(self):
        """Returns the flight path angle, in radians."""
        return np.arctan2(-self.w_e, self.u_e)

    def convert_axes(self,
                     x_from: float,
                     z_from: float,
                     from_axes: str,
                     to_axes: str,
                     ) -> Tuple[float, float]:
        if from_axes == "wind" or to_axes == "wind":
            sgam = np.sin(self.gamma)
            cgam = np.cos(self.gamma)

        if from_axes == "earth":
            x_e = x_from
            z_e = z_from
        elif from_axes == "wind":
            x_e = cgam * x_from + sgam * z_from
            z_e = -sgam * x_from + cgam * z_from
        else:
            raise ValueError("Bad value of `from_axes`!")

        if to_axes == "earth":
            x_to = x_e
            z_to = z_e
        elif to_axes == "wind":
            x_to = cgam * x_e - sgam * z_e
            z_to = sgam * x_e + cgam * z_e
        else:
            raise ValueError("Bad value of `to_axes`!")

        return x_to, z_to

    def add_force(self,
                  Fx: Union[np.ndarray, float] = 0,
                  Fz: Union[np.ndarray, float] = 0,
                  axes="wind",
                  ) -> None:
        Fx_e, Fz_e = self.convert_axes(
            x_from=Fx,
            z_from=Fz,
            from_axes=axes,
            to_axes="earth"
        )
        self.Fx_e = self.Fx_e + Fx_e
        self.Fz_e = self.Fz_e + Fz_e

    @property
    def speed(self) -> float:
        return (
                       self.u_e ** 2 +
                       self.z_e ** 2
               ) ** 0.5


if __name__ == '__main__':
    dyn = DynamicsPointMass2DCartesian()
