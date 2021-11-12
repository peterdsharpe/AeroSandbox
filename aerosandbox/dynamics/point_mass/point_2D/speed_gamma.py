from aerosandbox.dynamics.point_mass.common_point_mass import _DynamicsPointMassBaseClass
from aerosandbox.weights.mass_properties import MassProperties
import aerosandbox.numpy as np
from typing import Union, Dict, Tuple


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
            "Fx_w": self.Fx_w,
            "Fz_w": self.Fz_w,
        }

    def state_derivatives(self) -> Dict[str, Union[float, np.ndarray]]:
        return {
            "x_e"  : self.speed * np.cos(self.gamma),
            "z_e"  : -self.speed * np.sin(self.gamma),
            "speed": self.Fx_w / self.mass_props.mass,
            "gamma": -self.Fz_w / self.mass_props.mass / self.speed,
        }

    @property
    def speed(self) -> float:
        return self._speed

    @speed.setter
    def speed(self, value):
        self._speed = value

    @property
    def gamma(self) -> float:
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma = value

    def convert_axes(self,
                     x_from: float,
                     z_from: float,
                     from_axes: str,
                     to_axes: str,
                     ) -> Tuple[float, float]:
        if from_axes == "earth" or to_axes == "earth":
            sgam = np.sin(self.gamma)
            cgam = np.cos(self.gamma)

        if from_axes == "wind":
            x_w = x_from
            z_w = z_from
        elif from_axes == "earth":
            x_w = cgam * x_from - sgam * z_from
            z_w = sgam * x_from + cgam * z_from
        else:
            raise ValueError("Bad value of `from_axes`!")

        if to_axes == "wind":
            x_to = x_w
            z_to = z_w
        elif to_axes == "earth":
            x_to = cgam * x_w + sgam * z_w
            z_to = -sgam * x_w + cgam * z_w
        else:
            raise ValueError("Bad value of `to_axes`!")

        return x_to, z_to

    def add_force(self,
                  Fx: Union[np.ndarray, float] = 0,
                  Fz: Union[np.ndarray, float] = 0,
                  axes="wind",
                  ) -> None:
        Fx_w, Fz_w = self.convert_axes(
            x_from=Fx,
            z_from=Fz,
            from_axes=axes,
            to_axes="wind"
        )
        self.Fx_w = self.Fx_w + Fx_w
        self.Fz_w = self.Fz_w + Fz_w


if __name__ == '__main__':
    dyn = DynamicsPointMass2DSpeedGamma()
