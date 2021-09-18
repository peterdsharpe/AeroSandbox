from aerosandbox.common import *
import aerosandbox.numpy as np
from aerosandbox.dynamics.equations_of_motion import equations_of_motion
from aerosandbox import OperatingPoint, Atmosphere
import warnings


class FreeBodyDynamics(ImplicitAnalysis):
    @ImplicitAnalysis.initialize
    def __init__(self,
                 time: np.ndarray,
                 xe: np.ndarray = None,
                 ye: np.ndarray = None,
                 ze: np.ndarray = None,
                 u: np.ndarray = None,
                 v: np.ndarray = None,
                 w: np.ndarray = None,
                 phi: np.ndarray = None,
                 theta: np.ndarray = None,
                 psi: np.ndarray = None,
                 p: np.ndarray = None,
                 q: np.ndarray = None,
                 r: np.ndarray = None,
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
        self.xe = 0 if xe is None else xe
        self.ye = 0 if ye is None else ye
        self.ze = 0 if ze is None else ze
        self.u = 0 if u is None else u
        self.v = 0 if v is None else v
        self.w = 0 if w is None else w
        self.phi = 0 if phi is None else phi
        self.theta = 0 if theta is None else theta
        self.psi = 0 if psi is None else psi
        self.p = 0 if p is None else p
        self.q = 0 if q is None else q
        self.r = 0 if r is None else r
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
                warnings.warn(
                    "Can't add physics constraints to your dynamics environment, since no `opti` instance was provided. Skipping...")
            else:
                state = self.state
                state_derivatives = self.state_derivatives()
                for k in state.keys():  # TODO default to second-order integration for position, angles
                    if eval(k) is None: # Don't constrain states that haven't been defined by the user.
                        continue
                    try:
                        self.opti.constrain_derivative(
                            derivative=state_derivatives[k],
                            variable=state[k],
                            with_respect_to=self.time,
                        )
                    except Exception as e:
                        raise ValueError(f"Error while constraining state variable `{k}`: \n{e}")

    def __repr__(self):
        repr = []
        repr.append("Dynamics instance:")

        def trim(string, width=40):
            string = string.strip()
            if len(string) > width:
                return string[:width - 3] + "..."
            else:
                return string

        def makeline(k, v):
            name = trim(str(k), width=8).rjust(8)
            item = trim(str(v), width=40).ljust(40)

            try:
                value = str(self.opti.value(v))
            except:
                value = None

            if str(value).strip() == str(item).strip():
                value = None

            if isinstance(v, float) or isinstance(v, int) or isinstance(v, np.ndarray):
                value = None

            if value is not None:
                value = trim(value, width=40).ljust(40)
                return f"\t\t{name}: {item} ({value})"
            else:
                return f"\t\t{name}: {item}"

        repr.append("\tState variables:")
        for k, v in self.state.items():
            repr.append(makeline(k, v))

        repr.append("\tControl/other variables:")
        for k, v in self.control_variables.items():
            repr.append(makeline(k, v))

        return "\n".join(repr)

    # TODO add __getitem__ for dynamic state at instant in time

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
    def control_variables(self):
        return {
            "X"    : self.X,
            "Y"    : self.Y,
            "Z"    : self.Z,
            "L"    : self.L,
            "M"    : self.M,
            "N"    : self.N,
            "mass" : self.mass,
            "Ixx"  : self.Ixx,
            "Iyy"  : self.Iyy,
            "Izz"  : self.Izz,
            "Ixy"  : self.Ixy,
            "Iyz"  : self.Iyz,
            "Ixz"  : self.Ixz,
            "g"    : self.g,
            "hx"   : self.hx,
            "hy"   : self.hy,
            "hz"   : self.hz,
            "alpha": self.alpha,
            "beta" : self.beta,
            "speed": self.speed,
            "altitude": self.altitude
        }

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
    def altitude(self):
        return -self.ze

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

    n_timesteps = 300

    time = np.linspace(0, 1, n_timesteps)

    dyn = FreeBodyDynamics(
        opti=opti,
        time=time,
        xe=opti.variable(init_guess=np.linspace(0, 1, n_timesteps)),
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
