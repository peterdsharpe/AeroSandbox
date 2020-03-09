"""
Simple Beam

A simple 2D beam example, to be integrated later for full aerostructural modeling. TODO do that.

Governing equation:
Euler-Bernoulli beam theory.

(E * I * u(x)'')'' = q(x)

where:
    * E is the elastic modulus
    * I is the bending moment of inertia
    * u(x) is the local displacement at x.
    * q(x) is the force-per-unit-length at x. (In other words, a dirac delta is a point load.)
    * ()' is a derivative w.r.t. x.
"""
import numpy as np
import casadi as cas

opti = cas.Opti()

L = 40
n = 200
x = cas.linspace(0, L, n)
dx = cas.diff(x)
E = 228e9  # Pa, modulus of CF
max_allowable_stress = 570e6 / 1.75
# nominal_diameter = cas.linspace(0.2, 0.05, n)  # 0.1
nominal_diameter = opti.variable(n)
opti.set_initial(nominal_diameter, 100e-3)
opti.subject_to([
    nominal_diameter > 30e-3
])
thickness = 2e-3
I = cas.pi / 64 * ((nominal_diameter + thickness) ** 4 - (nominal_diameter - thickness) ** 4)
EI = E * I
q = cas.linspace(0, 0, n)
q[-1] = 700 / dx[-1]

u = opti.variable(n)
du = opti.variable(n)
ddu = opti.variable(n)
dEIddu = opti.variable(n)
opti.set_initial(u, 0)
opti.set_initial(du, 0)
opti.set_initial(ddu, 0)
opti.set_initial(dEIddu, 0)

# Add forcing term
ddEIddu = q


# Define derivatives
def trapz(x):
    out = (x[:-1] + x[1:]) / 2
    out[0] += x[0] / 2
    out[-1] += x[-1] / 2
    return out


opti.subject_to([
    cas.diff(u) == trapz(du) * dx,
    cas.diff(du) == trapz(ddu) * dx,
    cas.diff(EI * ddu) == trapz(dEIddu) * dx,
    cas.diff(dEIddu) == trapz(ddEIddu) * dx,
])

# Add BCs
opti.subject_to([
    u[0] == 0,
    du[0] == 0,
    ddu[-1] == 0,  # No tip moment
    dEIddu[-1] == 0,  # No tip higher order stuff
])

# Failure criterion
stress = (nominal_diameter + thickness) / 2 * E * ddu
opti.subject_to([
    stress < max_allowable_stress
])

# Mass
volume = cas.sum1(
    cas.pi / 4 * trapz((nominal_diameter + thickness) ** 2 - (nominal_diameter - thickness) ** 2) * dx
)
mass = volume * 1600
opti.minimize(mass)


p_opts = {}
s_opts = {}
s_opts["max_iter"] = 500  # If you need to interrupt, just use ctrl+c
# s_opts["mu_strategy"] = "adaptive"
opti.solver('ipopt', p_opts, s_opts)

try:
    sol = opti.solve()
except:
    print("Failed!")
    sol = opti.debug

import matplotlib.pyplot as plt
import matplotlib.style as style
import plotly.express as px
import plotly.graph_objects as go
import dash
import seaborn as sns

sns.set(font_scale=1)

fig, ax = plt.subplots(2, 3, figsize=(10, 6), dpi=200)

plt.subplot(231)
plt.plot(sol.value(x), sol.value(u), '.-')
plt.xlabel("x [m]")
plt.ylabel("u [m]")
plt.title("Displacement")

plt.subplot(232)
plt.plot(sol.value(x), sol.value(du), '.-')
plt.xlabel("x [m]")
plt.ylabel(r"$du/dx$ [rad]")
plt.title("Slope")

plt.subplot(233)
plt.plot(sol.value(x), sol.value(ddu), '.-')
plt.xlabel("x [m]")
plt.ylabel(r"$d^2u/dx^2$")
plt.title("Curvature (nondim. bending moment)")

plt.subplot(234)
plt.plot(sol.value(x), sol.value(stress / 1e6), '.-')
plt.xlabel("x [m]")
plt.ylabel("Stress [MPa]")
plt.title("Peak Stress at Section")

plt.subplot(235)
plt.plot(sol.value(x), sol.value(dEIddu), '.-')
plt.xlabel("x [m]")
plt.ylabel("u [m]")
plt.title("Shear Force")

plt.tight_layout()
# plt.legend()
plt.show()
