import aerosandbox as asb
from aerosandbox import cas
import numpy as np
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


def test_original_gpkit_like_solve():
    opti = asb.Opti()  # initialize an optimization environment

    ### Variables
    D = opti.variable(log_transform=True, init_guess=1000, scale=1e3)  # total drag force [N]
    A = opti.variable(log_transform=True, init_guess=10, scale=1e1)  # aspect ratio
    S = opti.variable(log_transform=True, init_guess=100, scale=1e2)  # total wing area [m^2]
    V = opti.variable(log_transform=True, init_guess=100, scale=1e2)  # cruising speed [m/s]
    W = opti.variable(log_transform=True, init_guess=8e3, scale=1e4)  # total aircraft weight [N]
    Re = opti.variable(log_transform=True, init_guess=5e6, scale=1e6)  # Reynolds number [-]
    C_D = opti.variable(log_transform=True, init_guess=0.03, scale=1e-2)  # Drag coefficient of wing [-]
    C_L = opti.variable(log_transform=True, init_guess=1, scale=1e-1)  # Lift coefficient of wing [-]
    C_f = opti.variable(log_transform=True, init_guess=0.01, scale=1e-2)  # Skin friction coefficient [-]
    W_w = opti.variable(log_transform=True, init_guess=3e3, scale=1e3)  # Wing weight [N]

    ### Constraints
    # Drag model
    C_D_fuse = CDA0 / S
    C_D_wpar = k * C_f * S_wetratio
    C_D_ind = C_L ** 2 / (pi * A * e)
    opti.subject_to(cas.log(C_D) >= cas.log(C_D_fuse + C_D_wpar + C_D_ind))

    # Wing weight model
    W_w_strc = W_W_coeff1 * (N_ult * A ** 1.5 * (W_0 * W * S) ** 0.5) / tau
    W_w_surf = W_W_coeff2 * S
    opti.subject_to(cas.log(W_w) >= cas.log(W_w_surf + W_w_strc))

    # Other models
    opti.subject_to([
        cas.log(D) >= cas.log(0.5 * rho * S * C_D * V ** 2),
        cas.log(Re) <= cas.log((rho / mu) * V * (S / A) ** 0.5),
        cas.log(C_f) >= cas.log(0.074 / Re ** 0.2),
        cas.log(W) <= cas.log(0.5 * rho * S * C_L * V ** 2),
        cas.log(W) <= cas.log(0.5 * rho * S * C_Lmax * V_min ** 2),
        cas.log(W) >= cas.log(W_0 + W_w),
    ])

    # Objective
    opti.minimize(cas.log(D))

    sol = opti.solve()

    assert sol.value(D) == pytest.approx(303.1, abs=0.1)


def test_rewritten_gpkit_like_solve():
    opti = asb.Opti()  # initialize an optimization environment

    ### Variables
    A = opti.variable(log_transform=True, init_guess=10, scale=1e1)  # aspect ratio
    S = opti.variable(log_transform=True, init_guess=100, scale=1e2)  # total wing area [m^2]
    V = opti.variable(log_transform=True, init_guess=100, scale=1e2)  # cruising speed [m/s]
    W = opti.variable(log_transform=True, init_guess=8e3, scale=1e4)  # total aircraft weight [N]
    C_L = opti.variable(log_transform=True, init_guess=1, scale=1e-1)  # Lift coefficient of wing [-]

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

    assert sol.value(D) == pytest.approx(303.1, abs=0.1)


def test_rewritten_gpkit_like_solve_no_guesses_or_scaling():
    opti = asb.Opti()  # initialize an optimization environment

    ### Variables
    A = opti.variable(log_transform=True)  # aspect ratio
    S = opti.variable(log_transform=True)  # total wing area [m^2]
    V = opti.variable(log_transform=True)  # cruising speed [m/s]
    W = opti.variable(log_transform=True)  # total aircraft weight [N]
    C_L = opti.variable(log_transform=True)  # Lift coefficient of wing [-]

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

    assert sol.value(D) == pytest.approx(303.1, abs=0.1)


def test_non_log_transformed_solve():
    opti = asb.Opti()  # initialize an optimization environment

    ### Variables
    A = opti.variable(init_guess=10, scale=1e1)  # aspect ratio
    S = opti.variable(init_guess=100, scale=1e2)  # total wing area [m^2]
    V = opti.variable(init_guess=100, scale=1e2)  # cruising speed [m/s]
    W = opti.variable(init_guess=8e3, scale=1e4)  # total aircraft weight [N]
    C_L = opti.variable(init_guess=1, scale=1e-1)  # Lift coefficient of wing [-]

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

    assert sol.value(D) == pytest.approx(303.1, abs=0.1)


if __name__ == '__main__':
    pytest.main()
