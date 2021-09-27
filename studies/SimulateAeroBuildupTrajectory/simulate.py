import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.dynamics.dynamics import FreeBodyDynamics
from conventional import airplane
from scipy import integrate

t_span = (0, 10)

def dynamics(t, y):
    dyn = FreeBodyDynamics(
        0,
        *y,
        Iyy=0.1,
        add_constraints=False
    )
    aero = asb.AeroBuildup(
        airplane=airplane,
        op_point=dyn.op_point
    ).run()
    dyn.X, dyn.Y, dyn.Z = aero["F_b"]
    dyn.L, dyn.M, dyn.N = aero["M_b"]
    return np.array(list(dyn.state_derivatives().values()))

init_state = {
    "xe"   : 0,
    "ye"   : 0,
    "ze"   : 0,
    "u"    : 10,
    "v"    : 0,
    "w"    : 0,
    "phi"  : 0,
    "theta": 0,
    "psi"  : 0,
    "p"    : 0,
    "q"    : 0,
    "r"    : 0,
}

print(dynamics(0, init_state.values()))

sol = integrate.solve_ivp(
    fun=dynamics,
    t_span=t_span,
    y0=np.array(list(init_state.values())),
    method="LSODA",
    vectorized=True
)
dyn = FreeBodyDynamics(
    sol.t,
    *sol.y,
    add_constraints=False
)
aero = asb.AeroBuildup(
    airplane=airplane,
    op_point=dyn.op_point
).run()
dyn.X, dyn.Y, dyn.Z = aero["F_b"]
dyn.L, dyn.M, dyn.N = aero["M_b"]

from aerosandbox.tools.pretty_plots import plt, show_plot
fig, ax = plt.subplots()
plt.plot(dyn.time, dyn.alpha)
plt.show()

