import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.dynamics.utilities import inv_symmetric_3x3
import warnings


def equations_of_motion(
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
):
    """
    Computes the state derivatives (i.e. equations of motion) for a body in 3D space.

    Based on Section 9.8.2 of Flight Vehicle Aerodynamics by Mark Drela.

    Args:
        xe: x-position, in Earth axes. [meters]
        ye: y-position, in Earth axes. [meters]
        ze: z-position, in Earth axes. [meters]
        u: x-velocity, in body axes. [m/s]
        v: y-velocity, in body axes. [m/s]
        w: z-velocity, in body axes. [m/s]
        phi: roll angle. Uses yaw-pitch-roll Euler angle convention. [rad]
        theta: pitch angle. Uses yaw-pitch-roll Euler angle convention. [rad]
        psi: yaw angle. Uses yaw-pitch-roll Euler angle convention. [rad]
        p: x-angular-velocity, in body axes. [rad/sec]
        q: y-angular-velocity, in body axes. [rad/sec]
        r: z-angular-velocity, in body axes. [rad/sec]
        X: x-direction force, in body axes. [N]
        Y: y-direction force, in body axes. [N]
        Z: z-direction force, in body axes. [N]
        L: Moment about the x axis, in body axes. Assumed these moments are applied about the center of mass. [Nm]
        M: Moment about the y axis, in body axes. Assumed these moments are applied about the center of mass. [Nm]
        N: Moment about the z axis, in body axes. Assumed these moments are applied about the center of mass. [Nm]
        mass: Mass of the body. [kg]
        Ixx: Respective component of the (symmetric) moment of inertia tensor.
        Iyy: Respective component of the (symmetric) moment of inertia tensor.
        Izz: Respective component of the (symmetric) moment of inertia tensor.
        Ixy: Respective component of the (symmetric) moment of inertia tensor.
        Ixz: Respective component of the (symmetric) moment of inertia tensor.
        Iyz: Respective component of the (symmetric) moment of inertia tensor.
        g: Magnitude of gravitational acceleration. Assumed to act in the positive-z-in-earth-axes ("downward") direction. [m/s^2]
        hx: x-component of onboard angular momentum (e.g. propellers), in body axes. [kg*m^2/sec]
        hy: y-component of onboard angular momentum (e.g. propellers), in body axes. [kg*m^2/sec]
        hz: z-component of onboard angular momentum (e.g. propellers), in body axes. [kg*m^2/sec]

    Returns:
        Time derivatives of each of the 12 state variables, given in a dictionary:
            {
                "xe"   : d_xe,
                "ye"   : d_ye,
                "ze"   : d_ze,
                "u"    : d_u,
                "v"    : d_v,
                "w"    : d_w,
                "phi"  : d_phi,
                "theta": d_theta,
                "psi"  : d_psi,
                "p"    : d_p,
                "q"    : d_q,
                "r"    : d_r,
            }
    """

    ### Trig Shorthands
    def sincos(x):
        try:
            x = np.mod(x, 2 * np.pi)
            one = np.ones_like(x)
            zero = np.zeros_like(x)

            if np.allclose(x, 0) or np.allclose(x, 2 * np.pi):
                sin = zero
                cos = one
            elif np.allclose(x, np.pi / 2):
                sin = one
                cos = zero
            elif np.allclose(x, np.pi):
                sin = zero
                cos = -one
            elif np.allclose(x, 3 * np.pi / 2):
                sin = -one
                cos = zero
            else:
                raise ValueError()
        except:
            sin = np.sin(x)
            cos = np.cos(x)
        return sin, cos

    # Do the trig
    sphi, cphi = sincos(phi)
    sthe, cthe = sincos(theta)
    spsi, cpsi = sincos(psi)

    ##### Equations of Motion

    ### Position derivatives
    d_xe = (
            (cthe * cpsi) * u +
            (sphi * sthe * cpsi - cphi * spsi) * v +
            (cphi * sthe * cpsi + sphi * spsi) * w
    )
    d_ye = (
            (cthe * sphi) * u +
            (sphi * sthe * spsi + cphi * cpsi) * v +
            (cphi * sthe * spsi - sphi * cpsi) * w
    )
    d_ze = (
            (-sthe) * u +
            (sphi * cthe) * v +
            (cphi * cthe) * w
    )
    ### Velocity derivatives
    d_u = (
            (X / mass - g * sthe) -
            q * w +
            r * v
    )
    d_v = (
            (Y / mass + g * sphi * cthe) -
            r * u +
            p * w
    )
    d_w = (
            (Z / mass + g * cphi * cthe) -
            p * v +
            q * u
    )
    ### Angle derivatives
    if np.any(cthe != 0):
        d_phi = (
                p +
                q * sphi * sthe / cthe +
                r * cphi * sthe / cthe
        )
    else:
        d_phi = 0

    d_theta = (
            q * cphi -
            r * sphi
    )

    if np.any(cthe != 0):
        d_psi = (
            q * sphi / cthe +
            r * cphi / cthe
        )
    else:
        d_psi = 0

    ### Angular velocity derivatives
    RHS_L = (
            L -
            (Izz - Iyy) * q * r -
            Iyz * (q ** 2 - r ** 2) -
            Ixz * p * q +
            Ixy * p * r -
            hz * q +
            hy * r
    )
    RHS_M = (
            M -
            (Ixx - Izz) * r * p -
            Ixz * (r ** 2 - p ** 2) -
            Ixy * q * r +
            Iyz * q * p -
            hx * r +
            hz * p
    )
    RHS_N = (
            N -
            (Iyy - Ixx) * p * q -
            Ixy * (p ** 2 - q ** 2) -
            Iyz * r * p +
            Ixz * r * q -
            hy * p +
            hx * q
    )
    i11, i22, i33, i12, i23, i13 = inv_symmetric_3x3(Ixx, Iyy, Izz, Ixy, Iyz, Ixz)

    d_p = i11 * RHS_L + i12 * RHS_M + i13 * RHS_N
    d_q = i12 * RHS_L + i22 * RHS_M + i23 * RHS_N
    d_r = i13 * RHS_L + i23 * RHS_M + i33 * RHS_N

    return {
        "xe"   : d_xe,
        "ye"   : d_ye,
        "ze"   : d_ze,
        "u"    : d_u,
        "v"    : d_v,
        "w"    : d_w,
        "phi"  : d_phi,
        "theta": d_theta,
        "psi"  : d_psi,
        "p"    : d_p,
        "q"    : d_q,
        "r"    : d_r,
    }
