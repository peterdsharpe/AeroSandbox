from aerosandbox.dynamics.point_mass.common_point_mass import _DynamicsPointMassBaseClass
from aerosandbox.weights.mass_properties import MassProperties
import aerosandbox.numpy as np
from typing import Union, Dict, Tuple


class DynamicsPointMass3DSpeedGammaTrack(_DynamicsPointMassBaseClass):
    """
    Dynamics instance:
    * simulating a point mass
    * in 3D
    * with velocity parameterized in speed-gamma-track space

    State variables:
        x_e: x-position, in Earth axes. [meters]
        y_e: y-position, in Earth axes. [meters]
        z_e: z-position, in Earth axes. [meters]
        speed: Speed; equivalent to u_w, the x-velocity in wind axes. [m/s]
        gamma: Flight path angle. [radians]
        track: Track angle. [radians]
            * Track of 0 == North == aligned with x_e axis
            * Track of np.pi / 2 == East == aligned with y_e axis

    Indirect control variables:
        alpha: Angle of attack. [degrees]
        beta: Sideslip angle. [degrees]
        bank: Bank angle. [radians]

    Control variables:
        Fx_w: Force along the wind-x axis. [N]
        Fy_w: Force along the wind-y axis. [N]
        Fz_w: Force along the wind-z axis. [N]

    """

    def __init__(self,
                 mass_props: MassProperties = None,
                 x_e: Union[float, np.ndarray] = 0,
                 y_e: Union[float, np.ndarray] = 0,
                 z_e: Union[float, np.ndarray] = 0,
                 speed: Union[float, np.ndarray] = 0,
                 gamma: Union[float, np.ndarray] = 0,
                 track: Union[float, np.ndarray] = 0,
                 alpha: Union[float, np.ndarray] = 0,
                 beta: Union[float, np.ndarray] = 0,
                 bank: Union[float, np.ndarray] = 0,
                 ):
        # Initialize state variables
        self.mass_props = MassProperties() if mass_props is None else mass_props
        self.x_e = x_e
        self.y_e = y_e
        self.z_e = z_e
        self.speed = speed
        self.gamma = gamma
        self.track = track

        # Initialize indirect control variables
        self.alpha = alpha
        self.beta = beta
        self.bank = bank

        # Initialize control variables
        self.Fx_w = 0
        self.Fy_w = 0
        self.Fz_w = 0

    @property
    def state(self) -> Dict[str, Union[float, np.ndarray]]:
        return {
            "x_e"  : self.x_e,
            "y_e"  : self.y_e,
            "z_e"  : self.z_e,
            "speed": self.speed,
            "gamma": self.gamma,
            "track": self.track,
        }

    @property
    def control_variables(self) -> Dict[str, Union[float, np.ndarray]]:
        return {
            "alpha": self.alpha,
            "beta" : self.beta,
            "bank" : self.bank,
            "Fx_w" : self.Fx_w,
            "Fy_w" : self.Fy_w,
            "Fz_w" : self.Fz_w,
        }

    def state_derivatives(self) -> Dict[str, Union[float, np.ndarray]]:

        d_speed = self.Fx_w / self.mass_props.mass

        sb = np.sin(self.bank)
        cb = np.cos(self.bank)

        force_gamma_direction = -cb * self.Fz_w - sb * self.Fy_w  # Force in the direction that acts to increase gamma
        force_track_direction = -sb * self.Fz_w + cb * self.Fy_w  # Force in the direction that acts to increase track

        d_gamma = force_gamma_direction / self.mass_props.mass / self.speed
        d_track = force_track_direction / self.mass_props.mass / self.speed / np.cos(self.gamma)

        return {
            "x_e"  : self.u_e,
            "y_e"  : self.v_e,
            "z_e"  : self.w_e,
            "speed": d_speed,
            "gamma": d_gamma,
            "track": d_track,
        }

    @property
    def u_e(self):
        return self.speed * np.cos(self.gamma) * np.cos(self.track)

    @property
    def v_e(self):
        return self.speed * np.cos(self.gamma) * np.sin(self.track)

    @property
    def w_e(self):
        return -self.speed * np.sin(self.gamma)

    def convert_axes(self,
                     x_from: float,
                     y_from: float,
                     z_from: float,
                     from_axes: str,
                     to_axes: str,
                     ) -> Tuple[float, float, float]:
        if from_axes == to_axes:
            return x_from, y_from, z_from

        if (from_axes == "earth" or to_axes == "earth"):
            rot_w_to_e = np.rotation_matrix_from_euler_angles(
                roll_angle=self.bank,
                pitch_angle=self.gamma,
                yaw_angle=self.track,
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
            x_w, y_w, z_w = self.op_point.convert_axes(
                x_from, y_from, z_from,
                from_axes=from_axes, to_axes="wind"
            )

        if to_axes == "wind":
            x_to = x_w
            y_to = y_w
            z_to = z_w
        elif to_axes == "earth":
            x_to = rot_w_to_e[0][0] * x_w + rot_w_to_e[0][1] * y_w + rot_w_to_e[0][2] * z_w
            y_to = rot_w_to_e[1][0] * x_w + rot_w_to_e[1][1] * y_w + rot_w_to_e[1][2] * z_w
            z_to = rot_w_to_e[2][0] * x_w + rot_w_to_e[2][1] * y_w + rot_w_to_e[2][2] * z_w
        else:
            x_to, y_to, z_to = self.op_point.convert_axes(
                x_w, y_w, z_w,
                from_axes="wind", to_axes=to_axes
            )

        return x_to, y_to, z_to

    def add_force(self,
                  Fx: Union[float, np.ndarray] = 0,
                  Fy: Union[float, np.ndarray] = 0,
                  Fz: Union[float, np.ndarray] = 0,
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
    dyn = DynamicsPointMass3DSpeedGammaTrack()
