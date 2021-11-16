from aerosandbox.dynamics.point_mass.common_point_mass import _DynamicsPointMassBaseClass
from aerosandbox.weights.mass_properties import MassProperties
import aerosandbox.numpy as np
from typing import Union, Dict, Tuple


class DynamicsPointMass3DSpeedGammaHeading(_DynamicsPointMassBaseClass):
    """
    Dynamics instance:
    * simulating a point mass
    * in 3D
    * with velocity parameterized in speed-gamma-heading space.
    * assuming coordinated flight (i.e., sideslip = 0, bank angle = 0)

    State variables:
        x_e: x-position, in Earth axes. [meters]
        y_e: y-position, in Earth axes. [meters]
        z_e: z-position, in Earth axes. [meters]
        speed: Speed; equivalent to u_w, the x-velocity in wind axes. [m/s]
        gamma: Flight path angle. [radians]
        heading: Heading angle. [radians]
            * Heading 0 == North == aligned with x_e axis
            * Heading np.pi / 2 == East == aligned with y_e axis
        bank: Bank angle; used when applying wind-axes forces. [radians] No derivative (assumed to be instantaneously reached)

    Control variables:
        Fx_w: Force along the wind-x axis. [N]
        Fy_w: Force along the wind-y axis. [N]
        Fz_w: Force along the wind-z axis. [N]
    """

    def __init__(self,
                 mass_props: MassProperties = None,
                 x_e: Union[np.ndarray, float] = 0,
                 y_e: Union[np.ndarray, float] = 0,
                 z_e: Union[np.ndarray, float] = 0,
                 speed: Union[np.ndarray, float] = 0,
                 gamma: Union[np.ndarray, float] = 0,
                 heading: Union[np.ndarray, float] = 0,
                 bank: Union[np.ndarray, float] = 0,
                 ):
        # Initialize state variables
        self.mass_props = MassProperties() if mass_props is None else mass_props
        self.x_e = x_e
        self.y_e = y_e
        self.z_e = z_e
        self.speed = speed
        self.gamma = gamma
        self.heading = heading
        self.bank = bank

        # Initialize control variables
        self.Fx_w = 0
        self.Fy_w = 0
        self.Fz_w = 0

    @property
    def state(self) -> Dict[str, Union[float, np.ndarray]]:
        return {
            "x_e"    : self.x_e,
            "y_e"    : self.y_e,
            "z_e"    : self.z_e,
            "speed"  : self.speed,
            "gamma"  : self.gamma,
            "heading": self.heading,
            "bank"   : self.bank,
        }

    @property
    def control_variables(self) -> Dict[str, Union[float, np.ndarray]]:
        return {
            "Fx_w": self.Fx_w,
            "Fy_w": self.Fy_w,
            "Fz_w": self.Fz_w,
        }

    def state_derivatives(self) -> Dict[str, Union[float, np.ndarray]]:

        d_speed = self.Fx_w / self.mass_props.mass

        sb = np.sin(self.bank)
        cb = np.cos(self.bank)

        force_gamma_direction = -cb * self.Fz_w - sb * self.Fy_w  # Force in the direction that acts to increase gamma
        force_heading_direction = -sb * self.Fz_w + cb * self.Fy_w  # Force in the direction that acts to increase heading

        d_gamma = force_gamma_direction / self.mass_props.mass / self.speed
        d_heading = force_heading_direction / self.mass_props.mass / self.speed / np.cos(self.gamma)

        return {
            "x_e"    : self.u_e,
            "y_e"    : self.v_e,
            "z_e"    : self.w_e,
            "speed"  : d_speed,
            "gamma"  : d_gamma,
            "heading": d_heading,
            "bank"   : None,
        }

    @property
    def u_e(self):
        return self.speed * np.cos(self.gamma) * np.cos(self.heading)

    @property
    def v_e(self):
        return self.speed * np.cos(self.gamma) * np.sin(self.heading)

    @property
    def w_e(self):
        return -self.speed * np.sin(self.gamma)

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
                     y_from: float,
                     z_from: float,
                     from_axes: str,
                     to_axes: str,
                     ) -> Tuple[float, float, float]:
        if from_axes == "earth" or to_axes == "earth":
            rot_w_to_e = np.rotation_matrix_from_euler_angles(
                roll_angle=self.bank,
                pitch_angle=self.gamma,
                yaw_angle=self.heading,
                as_array=False
            )

        if from_axes == "wind":
            x_w = x_from
            y_w = y_from
            z_w = z_from
        elif from_axes == "earth":
            x_w = rot_w_to_e[0][0] * x_from + rot_w_to_e[1][0] * y_from + rot_w_to_e[2][0] * z_from
            y_w = rot_w_to_e[0][1] * x_from + rot_w_to_e[1][1] * y_from + rot_w_to_e[2][1] * z_from
            z_w = rot_w_to_e[0][2] * x_from + rot_w_to_e[1][2] * y_from + rot_w_to_e[2][2] * z_from
        else:
            raise ValueError("Bad value of `from_axes`!")

        if to_axes == "wind":
            x_to = x_w
            y_to = y_w
            z_to = z_w
        elif to_axes == "earth":
            x_to = rot_w_to_e[0][0] * x_w + rot_w_to_e[0][1] * y_w + rot_w_to_e[0][2] * z_w
            y_to = rot_w_to_e[1][0] * x_w + rot_w_to_e[1][1] * y_w + rot_w_to_e[1][2] * z_w
            z_to = rot_w_to_e[2][0] * x_w + rot_w_to_e[2][1] * y_w + rot_w_to_e[2][2] * z_w
        else:
            raise ValueError("Bad value of `to_axes`!")

        return x_to, y_to, z_to

    def add_force(self,
                  Fx: Union[np.ndarray, float] = 0,
                  Fy: Union[np.ndarray, float] = 0,
                  Fz: Union[np.ndarray, float] = 0,
                  axes="wind",
                  ) -> None:
        Fx_w, Fy_w, Fz_w = self.convert_axes(
            x_from=Fx,
            y_from=Fy,
            z_from=Fz,
            from_axes=axes,
            to_axes="wind"
        )
        self.Fx_w = self.Fx_w + Fx_w
        self.Fy_w = self.Fy_w + Fy_w
        self.Fz_w = self.Fz_w + Fz_w


if __name__ == '__main__':
    dyn = DynamicsPointMass3DSpeedGammaHeading()
