import aerosandbox as asb
import aerosandbox.numpy as np


def inv_symmetric_3x3(
        m11,
        m22,
        m33,
        m12,
        m23,
        m13,
):
    """
    Computes the inverse of a symmetric 3x3 matrix.

    From https://math.stackexchange.com/questions/233378/inverse-of-a-3-x-3-covariance-matrix-or-any-positive-definite-pd-matrix



    """
    det = (
            m11 * (m33 * m22 - m23 ** 2) -
            m12 * (m33 * m12 - m23 * m13) +
            m13 * (m23 * m12 - m22 * m13)
    )
    a11 = m33 * m22 - m23 ** 2
    a12 = m13 * m23 - m33 * m12
    a13 = m12 * m23 - m13 * m22

    a22 = m33 * m11 - m13 ** 2
    a23 = m12 * m13 - m11 * m23

    a33 = m11 * m22 - m12 ** 2

    a11 = a11 / det
    a12 = a12 / det
    a13 = a13 / det
    a22 = a22 / det
    a23 = a23 / det
    a33 = a33 / det

    return a11, a22, a33, a12, a23, a13


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
        g=9.81,
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
        Time derivatives of each of the 12 state variables, given in the following order:
            d_xe,
            d_ye,
            d_ze,
            d_u,
            d_v,
            d_w,
            d_phi,
            d_theta,
            d_psi,
            d_p,
            d_q,
            d_r,
            d_X,
            d_Y,
            d_Z,
            d_L,
            d_M,
            d_N,

    """

    ### Trig Shorthands
    sphi = np.sin(phi)
    cphi = np.cos(phi)
    sthe = np.sin(theta)
    cthe = np.cos(theta)
    spsi = np.sin(psi)
    cpsi = np.cos(psi)

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
    d_phi = (
            p +
            q * sphi * sthe / cthe +
            r * cphi * sthe / cthe
    )
    d_theta = (
            q * cphi -
            r * sphi
    )
    d_psi = (
            q * sphi / cthe +
            r * cphi / cthe
    )
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

    return d_xe, d_ye, d_ze, d_u, d_v, d_w, d_phi, d_theta, d_psi, d_p, d_q, d_r
