from aerosandbox.dynamics.rigid_body.rigid_3D.body_euler import DynamicsRigidBody3DBodyEuler
from aerosandbox.weights.mass_properties import MassProperties
import aerosandbox.numpy as np
from typing import Union, Dict, Tuple


class DynamicsRigidBody2DBody(DynamicsRigidBody3DBodyEuler):
    """
    Dynamics instance:
    * simulating a rigid body
    * in 2D
    * with velocity parameterized in body axes

    State variables:
        x_e: x-position, in Earth axes. [meters]
        z_e: z-position, in Earth axes. [meters]
        u_b: x-velocity, in body axes. [m/s]
        w_b: z-velocity, in body axes. [m/s]
        theta: pitch angle. [rad]
        q: y-angular-velocity, in body axes. [rad/sec]

    Control variables:
        Fx_b: Force along the body-x axis. [N]
        Fz_b: Force along the body-z axis. [N]
        My_b: Moment about the body-y axis. [Nm]

    """

    def __init__(self,
                 mass_props: MassProperties = None,
                 x_e: Union[float, np.ndarray] = 0,
                 z_e: Union[float, np.ndarray] = 0,
                 u_b: Union[float, np.ndarray] = 0,
                 w_b: Union[float, np.ndarray] = 0,
                 theta: Union[float, np.ndarray] = 0,
                 q: Union[float, np.ndarray] = 0,
                 ):
        # Initialize state variables
        self.mass_props = MassProperties() if mass_props is None else mass_props
        self.x_e = x_e
        self.y_e = 0
        self.z_e = z_e
        self.u_b = u_b
        self.v_b = 0
        self.w_b = w_b
        self.phi = 0
        self.theta = theta
        self.psi = 0
        self.p = 0
        self.q = q
        self.r = 0

        # Initialize control variables
        self.Fx_b = 0
        self.Fy_b = 0
        self.Fz_b = 0
        self.Mx_b = 0
        self.My_b = 0
        self.Mz_b = 0
        self.hx_b = 0
        self.hy_b = 0
        self.hz_b = 0

    @property
    def state(self):
        return {
            "x_e"  : self.x_e,
            "z_e"  : self.z_e,
            "u_b"  : self.u_b,
            "w_b"  : self.w_b,
            "theta": self.theta,
            "q"    : self.q,
        }

    @property
    def control_variables(self):
        return {
            "Fx_b": self.Fx_b,
            "Fz_b": self.Fz_b,
            "My_b": self.My_b,
        }

    def state_derivatives(self) -> Dict[str, Union[float, np.ndarray]]:
        derivatives = super().state_derivatives()
        return {
            k: derivatives[k] for k in self.state.keys()
        }


if __name__ == '__main__':
    dyn = DynamicsRigidBody2DBody()
