from aerosandbox.common import *
import aerosandbox.numpy as np
from aerosandbox.dynamics.equations_of_motion import equations_of_motion
from aerosandbox import OperatingPoint, Atmosphere
import warnings

class FreeBodyDynamics(ImplicitAnalysis):
    @ImplicitAnalysis.initialize
    def __init__(self,
                 time: np.ndarray,
                 xe=0,
                 ye=0,
                 ze=0,
                 u=0,
                 v=0,
                 w=0,
                 phi=0,
                 theta=0,
                 psi=0,
                 p=0,
                 q=0,
                 r=0,
                 X=0,
                 Y=0,
                 Z=0,
                 L=0,
                 M=0,
                 N=0,
                 mass=1,
                 Ixx=1,
                 Iyy=1,
                 Izz=1,
                 Ixy=0,
                 Iyz=0,
                 Ixz=0,
                 g=0,
                 hx=0,
                 hy=0,
                 hz=0,
                 add_constraints: bool = True,
                 ):

        self.time = time
        self.xe = xe
        self.ye = ye
        self.ze = ze
        self.u = u
        self.v = v
        self.w = w
        self.phi = phi
        self.theta = theta
        self.psi = psi
        self.p = p
        self.q = q
        self.r = r
        self.X = X
        self.Y = Y
        self.Z = Z
        self.L = L
        self.M = M
        self.N = N
        self.mass = mass
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz
        self.Ixy = Ixy
        self.Iyz = Iyz
        self.Ixz = Ixz
        self.g = g
        self.hx = hx
        self.hy = hy
        self.hz = hz

        if add_constraints:
            if not self.opti_provided:
                warnings.warn("Can't add physics constraints to your dynamics environment, since no `opti` instance was provided. Skipping...")
            else:
                state = self.state
                state_derivatives = self.state_derivatives()
                for k in state.keys(): # TODO default to second-order integration for position, angles
                    self.opti.constrain_derivative(
                        derivative=state_derivatives[k],
                        variable=state[k],
                        with_respect_to=self.time,
                    )

    @property
    def state(self):
        return {
            "xe"   : self.xe,
            "ye"   : self.ye,
            "ze"   : self.ze,
            "u"    : self.u,
            "v"    : self.v,
            "w"    : self.w,
            "phi"  : self.phi,
            "theta": self.theta,
            "psi"  : self.psi,
            "p"    : self.p,
            "q"    : self.q,
            "r"    : self.r,
        }

    def state_derivatives(self):
        return equations_of_motion(
            xe=self.xe,
            ye=self.ye,
            ze=self.ze,
            u=self.u,
            v=self.v,
            w=self.w,
            phi=self.phi,
            theta=self.theta,
            psi=self.psi,
            p=self.p,
            q=self.q,
            r=self.r,
            X=self.X,
            Y=self.Y,
            Z=self.Z,
            L=self.L,
            M=self.M,
            N=self.N,
            mass=self.mass,
            Ixx=self.Ixx,
            Iyy=self.Iyy,
            Izz=self.Izz,
            Ixy=self.Ixy,
            Iyz=self.Iyz,
            Ixz=self.Ixz,
            g=self.g,
            hx=self.hx,
            hy=self.hy,
            hz=self.hz,
        )

    @property
    def alpha(self):
        """The angle of attack, in degrees."""
        return np.arctan2d(
            -self.w,
            self.u
        )

    @property
    def beta(self):
        """The sideslip angle, in degrees."""
        return np.arctan2d(
            self.v,
            (self.u ** 2 + self.w ** 2) ** 0.5
        )

    @property
    def speed(self):
        """The speed of the object, expressed as a scalar."""
        return (
            self.u ** 2 +
            self.v ** 2 +
            self.w ** 2
        ) ** 0.5

    @property
    def op_point(self):
        return OperatingPoint(
            atmosphere=Atmosphere(altitude=-self.ze),
            velocity=self.speed,
            alpha=self.alpha,
            beta=self.beta,
            p=self.p,
            q=self.q,
            r=self.r,
        )


if __name__ == '__main__':
    import aerosandbox as asb

    opti = asb.Opti()

    n_timesteps=300

    time = np.linspace(0, 1, n_timesteps)

    dyn = FreeBodyDynamics(
        opti = opti,
        time = time,
        xe = opti.variable(init_guess=np.linspace(0, 1, n_timesteps)),
        u=opti.variable(init_guess=1, n_vars=n_timesteps),
        X=opti.variable(init_guess=np.linspace(1, -1, n_timesteps)),
    )

    opti.subject_to([
        dyn.xe[0] == 0,
        dyn.xe[-1] == 1,
        dyn.u[0] == 0,
        dyn.u[-1] == 0,
    ])

    opti.minimize(
        np.sum(np.trapz(dyn.X ** 2) * np.diff(time))
    )

    sol = opti.solve()

    dyn.substitute_solution(sol)