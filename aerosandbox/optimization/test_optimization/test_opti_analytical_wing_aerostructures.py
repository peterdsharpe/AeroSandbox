import aerosandbox as asb
import aerosandbox.numpy as np
import pytest

"""
These tests all implement a simple wing aerostructural optimization problem.

Problem is taken from Section 3 of "Geometric Programming for Aircraft Design Optimization" by W. Hoburg and P. Abbeel.
http://web.mit.edu/~whoburg/www/papers/hoburgabbeel2014.pdf

GPKit implementation available at: https://gpkit.readthedocs.io/en/latest/examples.html#simple-wing

Each test solves the same problem in a slightly different way, verifying the correct solution is obtained for all flags.

"""
from numpy import pi

### Constants
k = 1.2  # form factor [-]
e = 0.95  # Oswald efficiency factor [-]
mu = 1.78e-5  # viscosity of air [kg/m/s]
rho = 1.23  # density of air [kg/m^3]
tau = 0.12  # airfoil thickness to chord ratio [-]
N_ult = 3.8  # ultimate load factor [-]
V_min = 22  # takeoff speed [m/s]
C_Lmax = 1.5  # max CL with flaps down [-]
S_wetratio = 2.05  # wetted area ratio [-]
W_W_coeff1 = 8.71e-5  # Wing Weight Coefficient 1 [1/m]
W_W_coeff2 = 45.24  # Wing Weight Coefficient 2 [Pa]
CDA0 = 0.031  # fuselage drag area [m^2]
W_0 = 4940.0  # aircraft weight excluding wing [N]


def test_gpkit_style_solve():
    """
    Here, the problem is formulated *exactly* as it is in the GPKit paper.

    This isn't how you would actually want to solve this problem (lots of redundant variables here...), but
    this test just confirms that it's a valid mathematical formulation and still works.

    Also note that all constraints and the objective are log-transformed, so under the hood, this is basically
    exactly a geometric program (and should be convex, yay).
    """

    opti = asb.Opti()  # initialize an optimization environment

    ### Variables
    D = opti.variable(init_guess=1e3, log_transform=True)  # total drag force [N]
    A = opti.variable(init_guess=1e1, log_transform=True)  # aspect ratio
    S = opti.variable(init_guess=1e2, log_transform=True)  # total wing area [m^2]
    V = opti.variable(init_guess=1e2, log_transform=True)  # cruising speed [m/s]
    W = opti.variable(init_guess=8e3, log_transform=True)  # total aircraft weight [N]
    Re = opti.variable(init_guess=5e6, log_transform=True)  # Reynolds number [-]
    C_D = opti.variable(init_guess=3e-2, log_transform=True)  # Drag coefficient of wing [-]
    C_L = opti.variable(init_guess=1, log_transform=True)  # Lift coefficient of wing [-]
    C_f = opti.variable(init_guess=1e-2, log_transform=True)  # Skin friction coefficient [-]
    W_w = opti.variable(init_guess=3e3, log_transform=True)  # Wing weight [N]

    ### Constraints
    # Drag model
    C_D_fuse = CDA0 / S
    C_D_wpar = k * C_f * S_wetratio
    C_D_ind = C_L ** 2 / (pi * A * e)
    opti.subject_to(np.log(C_D) >= np.log(C_D_fuse + C_D_wpar + C_D_ind))

    # Wing weight model
    W_w_strc = W_W_coeff1 * (N_ult * A ** 1.5 * (W_0 * W * S) ** 0.5) / tau
    W_w_surf = W_W_coeff2 * S
    opti.subject_to(np.log(W_w) >= np.log(W_w_surf + W_w_strc))

    # Other models
    opti.subject_to([
        np.log(D) >= np.log(0.5 * rho * S * C_D * V ** 2),
        np.log(Re) <= np.log((rho / mu) * V * (S / A) ** 0.5),
        np.log(C_f) >= np.log(0.074 / Re ** 0.2),
        np.log(W) <= np.log(0.5 * rho * S * C_L * V ** 2),
        np.log(W) <= np.log(0.5 * rho * S * C_Lmax * V_min ** 2),
        np.log(W) >= np.log(W_0 + W_w),
    ])

    # Objective
    opti.minimize(np.log(D))

    sol = opti.solve()

    assert sol(D) == pytest.approx(303.1, abs=0.1)


def test_geometric_program_solve():
    """
    This still solves the problem like a geometric program (at least in the sense that the variables are
    log-transformed; note that the constraints aren't so it's not a true geometric program), but it removes redundant
    variables where possible. Basically you only need to add design variables for implicit things like total aircraft
    weight; no need to add unnecessary variables for explicit things like wing weight.
    """
    opti = asb.Opti()  # initialize an optimization environment

    ### Variables
    A = opti.variable(init_guess=10, log_transform=True)  # aspect ratio
    S = opti.variable(init_guess=100, log_transform=True)  # total wing area [m^2]
    V = opti.variable(init_guess=100, log_transform=True)  # cruising speed [m/s]
    W = opti.variable(init_guess=8e3, log_transform=True)  # total aircraft weight [N]
    C_L = opti.variable(init_guess=1, log_transform=True)  # Lift coefficient of wing [-]

    ### Constraints
    # Aerodynamics model
    C_D_fuse = CDA0 / S
    Re = (rho / mu) * V * (S / A) ** 0.5
    C_f = 0.074 / Re ** 0.2
    C_D_wpar = k * C_f * S_wetratio
    C_D_ind = C_L ** 2 / (pi * A * e)
    C_D = C_D_fuse + C_D_wpar + C_D_ind
    q = 0.5 * rho * V ** 2
    D = q * S * C_D
    L_cruise = q * S * C_L
    L_takeoff = 0.5 * rho * S * C_Lmax * V_min ** 2

    # Wing weight model
    W_w_strc = W_W_coeff1 * (N_ult * A ** 1.5 * (W_0 * W * S) ** 0.5) / tau
    W_w_surf = W_W_coeff2 * S
    W_w = W_w_surf + W_w_strc

    # Other constraints
    opti.subject_to([
        W <= L_cruise,
        W <= L_takeoff,
        W >= W_0 + W_w
    ])

    # Objective
    opti.minimize(D)

    sol = opti.solve()

    assert sol(D) == pytest.approx(303.1, abs=0.1)


def test_non_log_transformed_solve():
    """
    This is how you would solve the problem without any kind of log-transformation, and honestly, probably how you
    should solve this problem. Yes, log-transforming things into a geometric program makes everything convex,
    which is nice - however, even non-transformed, the problem is unimodal so IPOPT will find the global minimum.

    Forgoing the log-transform also makes the backend math faster, so it's kind of a trade-off. I tend to lean
    towards not log-transforming things unless they'd really cause stuff to blow up (for example, if you have some
    dynamic system where F=ma and mass ever goes negative, things will explode, so I might log-transform mass).

    Anyway, this should solve and get the same solution as the other problem formulations in this test file.
    """
    opti = asb.Opti()  # initialize an optimization environment

    ### Variables
    A = opti.variable(init_guess=10)  # aspect ratio
    S = opti.variable(init_guess=100)  # total wing area [m^2]
    V = opti.variable(init_guess=100)  # cruising speed [m/s]
    W = opti.variable(init_guess=8e3)  # total aircraft weight [N]
    C_L = opti.variable(init_guess=1)  # Lift coefficient of wing [-]

    ### Constraints
    # Aerodynamics model
    C_D_fuse = CDA0 / S
    Re = (rho / mu) * V * (S / A) ** 0.5
    C_f = 0.074 / Re ** 0.2
    C_D_wpar = k * C_f * S_wetratio
    C_D_ind = C_L ** 2 / (pi * A * e)
    C_D = C_D_fuse + C_D_wpar + C_D_ind
    q = 0.5 * rho * V ** 2
    D = q * S * C_D
    L_cruise = q * S * C_L
    L_takeoff = 0.5 * rho * S * C_Lmax * V_min ** 2

    # Wing weight model
    W_w_strc = W_W_coeff1 * (N_ult * A ** 1.5 * (W_0 * W * S) ** 0.5) / tau
    W_w_surf = W_W_coeff2 * S
    W_w = W_w_surf + W_w_strc

    # Other constraints
    opti.subject_to([
        W <= L_cruise,
        W <= L_takeoff,
        W == W_0 + W_w
    ])

    # Objective
    opti.minimize(D)

    sol = opti.solve()

    assert sol(D) == pytest.approx(303.1, abs=0.1)


if __name__ == '__main__':
    pytest.main()
