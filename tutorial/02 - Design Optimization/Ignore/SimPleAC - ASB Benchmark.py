import aerosandbox as asb
import aerosandbox.numpy as np
import time

def solve():
    opti = asb.Opti()

    ### Env. constants
    g = 9.81  # gravitational acceleration, m/s^2
    mu = 1.775e-5  # viscosity of air, kg/m/s
    rho = 1.23  # density of air, kg/m^3
    rho_f = 817  # density of fuel, kg/m^3

    ### Non-dimensional constants
    C_Lmax = 1.6  # stall CL
    e = 0.92  # Oswald's efficiency factor
    k = 1.17  # form factor
    N_ult = 3.3  # ultimate load factor
    S_wetratio = 2.075  # wetted area ratio
    tau = 0.12  # airfoil thickness to chord ratio
    W_W_coeff1 = 2e-5  # wing weight coefficient 1
    W_W_coeff2 = 60  # wing weight coefficient 2

    ### Dimensional constants
    Range = 1000e3  # aircraft range, m
    TSFC = 0.6 / 3600  # thrust specific fuel consumption, 1/sec
    V_min = 25  # takeoff speed, m/s
    W_0 = 6250  # aircraft weight excluding wing, N

    ### Free variables (same as SimPleAC, with extraneous variables removed)
    AR = opti.variable(init_guess=10, log_transform=True)  # aspect ratio
    S = opti.variable(init_guess=10, log_transform=True)  # total wing area, m^2
    V = opti.variable(init_guess=100, log_transform=True)  # cruise speed, m/s
    W = opti.variable(init_guess=10000, log_transform=True)  # total aircraft weight, N
    C_L = opti.variable(init_guess=1, log_transform=True)  # lift coefficient
    W_f = opti.variable(init_guess=3000, log_transform=True)  # fuel weight, N
    V_f_fuse = opti.variable(init_guess=1, log_transform=True)  # fuel volume in the fuselage, m^3

    ### Wing weight
    W_w_surf = W_W_coeff2 * S
    W_w_strc = W_W_coeff1 / tau * N_ult * AR ** 1.5 * np.sqrt(
        (W_0 + V_f_fuse * g * rho_f) * W * S
    )
    W_w = W_w_surf + W_w_strc

    ### Entire weight
    opti.subject_to(
        W >= W_0 + W_w + W_f
    )

    ### Lift equals weight constraint
    opti.subject_to([
        W_0 + W_w + 0.5 * W_f <= 0.5 * rho * S * C_L * V ** 2,
        W <= 0.5 * rho * S * C_Lmax * V_min ** 2,
    ])

    ### Flight duration
    T_flight = Range / V

    ### Drag
    Re = (rho / mu) * V * (S / AR) ** 0.5
    C_f = 0.074 / Re ** 0.2

    CDA0 = V_f_fuse / 10

    C_D_fuse = CDA0 / S
    C_D_wpar = k * C_f * S_wetratio
    C_D_ind = C_L ** 2 / (np.pi * AR * e)
    C_D = C_D_fuse + C_D_wpar + C_D_ind
    D = 0.5 * rho * S * C_D * V ** 2

    opti.subject_to([
        W_f >= TSFC * T_flight * D,
    ])

    V_f = W_f / g / rho_f
    V_f_wing = 0.03 * S ** 1.5 / AR ** 0.5 * tau  # linear with b and tau, quadratic with chord

    V_f_avail = V_f_wing + V_f_fuse

    opti.subject_to(
        V_f_avail >= V_f
    )

    opti.minimize(W_f)

    sol = opti.solve(verbose=False)

def timeit():
    start = time.time()
    solve()
    end = time.time()
    return end - start

if __name__ == '__main__':
    times = np.array([
        timeit() for i in range(10)
    ])
    print(np.mean(times))