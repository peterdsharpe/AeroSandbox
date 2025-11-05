import aerosandbox as asb
import aerosandbox.numpy as np

N = 500  # Number of discretization nodes
E = 1e3  # Elastic modulus [N/m^2]
L = 1  # Beam length [m]
b = 0.1  # Beam width [m]
volume = 0.01  # Total material allowed [m^3]
tip_load = 1  # Tip load [N]

x = np.linspace(0, L, N)  # Node locations [m]

opti = asb.Opti()
h = opti.variable(init_guess=np.ones(N), lower_bound=1e-6)
I = (1 / 12) * b * h**3  # Bending moment of inertia [m^4]

V = np.ones(N) * (-tip_load)  # Shear force [N]
M = opti.variable(init_guess=np.zeros(N))  # Moment [N*m]
th = opti.variable(init_guess=np.zeros(N))  # Slope [rad]
w = opti.variable(init_guess=np.zeros(N))  # Displacement [m]

opti.subject_to(
    [  # Governing equations
        np.diff(M) == np.trapz(V) * np.diff(x),
        np.diff(th) == np.trapz(M / (E * I), modify_endpoints=True) * np.diff(x),
        np.diff(w) == np.trapz(th) * np.diff(x),
    ]
)
opti.subject_to(
    [  # Boundary conditions
        M[-1] == 0,
        th[0] == 0,
        w[0] == 0,
    ]
)
opti.subject_to(np.mean(h * b) <= volume / L)  # Volume constraint
opti.minimize(w[-1])  # Objective: minimize tip deflection
sol = opti.solve()
print(sol(h))  # Gives the optimized beam thickness

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    plt.plot(x, sol(h), label="ASB")
    p.show_plot("ASB")
