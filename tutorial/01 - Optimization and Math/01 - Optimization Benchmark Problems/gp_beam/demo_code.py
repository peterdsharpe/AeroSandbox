import aerosandbox as asb
import aerosandbox.numpy as np

N = 50 # Number of discretization nodes
L = 6  # m, overall beam length
EI = 1.1e4  # N*m^2, bending stiffness
q = 110 * np.ones(N)  # N/m, distributed load

x = np.linspace(0, L, N)  # m, node locations

opti = asb.Opti()  # set up an optimization environment

w = opti.variable(init_guess=np.zeros(N))  # m, displacement

th = opti.derivative_of(  # rad, slope
    w, with_respect_to=x,
    derivative_init_guess=np.zeros(N),
)

M = opti.derivative_of(  # N*m, moment
    th * EI, with_respect_to=x,
    derivative_init_guess=np.zeros(N),
)

V = opti.derivative_of(  # N, shear
    M, with_respect_to=x,
    derivative_init_guess=np.zeros(N),
)

opti.constrain_derivative(
    variable=V, with_respect_to=x,
    derivative=q,
)

opti.subject_to([  # Boundary conditions
    w[0] == 0,
    th[0] == 0,
    M[-1] == 0,
    V[-1] == 0,
])

sol = opti.solve()

print(sol(w[-1]))  # Prints the tip deflection; should be 1.62 m.
