import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.dynamics.dynamics import FreeBodyDynamics
from conventional import airplane
from scipy import integrate


t_span = (0, 120)


def dynamics(t, y):
    dyn = FreeBodyDynamics(
        0,
        *y,
        g=1,
        add_constraints=False
    )
    aero = asb.AeroBuildup(
        airplane=airplane,
        op_point=dyn.op_point
    ).run()
    dyn.X, dyn.Y, dyn.Z = aero["F_b"]
    dyn.L, dyn.M, dyn.N = aero["M_b"]
    derivatives = dyn.state_derivatives()
    derivatives["u"] = np.where(
        dyn.u < 0,
        0,
        derivatives["u"]
    )

    return np.array(list(dyn.state_derivatives().values()))


init_state = {
    "xe"   : 0,
    "ye"   : 0,
    "ze"   : 0,
    "u"    : 5,
    "v"    : 0,
    "w"    : 0,
    "phi"  : 0,
    "theta": 0,
    "psi"  : 0,
    "p"    : 0,
    "q"    : 0,
    "r"    : 0,
}

atols = {
    "xe"   : 1,
    "ye"   : 1,
    "ze"   : 1,
    "u"    : 0.1,
    "v"    : 0.1,
    "w"    : 0.1,
    "phi"  : 0.01,
    "theta": 0.01,
    "psi"  : 0.01,
    "p"    : 0.01,
    "q"    : 0.01,
    "r"    : 0.01,
}

sol = integrate.solve_ivp(
    fun=dynamics,
    t_span=t_span,
    y0=np.array(list(init_state.values())),
    method="LSODA",
    vectorized=True,
    dense_output=True,
    atol=np.array(list(atols.values())),
)
time = np.linspace(sol.t.min(), sol.t.max(), 300)
dyn = FreeBodyDynamics(
    time,
    *sol.sol(time),
    add_constraints=False
)
aero = asb.AeroBuildup(
    airplane=airplane,
    op_point=dyn.op_point
).run()
dyn.X, dyn.Y, dyn.Z = aero["F_b"]
dyn.L, dyn.M, dyn.N = aero["M_b"]

from aerosandbox.tools.pretty_plots import plt, show_plot, equal

vars_to_plot = {
    **dyn.state,
    "alpha"   : dyn.alpha,
    "beta"    : dyn.beta,
    "speed"   : dyn.speed,
    "altitude": dyn.altitude,
    "CL"      : aero["CL"],
    "CY"      : aero["CY"],
    "CD"      : aero["CD"],
    "Cl"      : aero["Cl"],
    "Cm"      : aero["Cm"],
    "Cn"      : aero["Cn"],
}
fig, axes = plt.subplots(6, 4, figsize=(15, 10), sharex=True)
for var_to_plot, ax in zip(vars_to_plot.items(), axes.flatten(order="F")):
    plt.sca(ax)
    k, v = var_to_plot
    plt.plot(dyn.time, v)
    plt.ylabel(k)
show_plot()

fig, ax = plt.subplots()
plt.plot(dyn.xe, dyn.altitude, "k")
sc =plt.scatter(dyn.xe, dyn.altitude, c=dyn.speed, cmap=plt.get_cmap("rainbow"), zorder=4)
plt.axis('equal')
plt.colorbar(label="Airspeed [m/s]")
show_plot("Trajectory using `asb.AeroBuildup` Flight Dynamics", "$x_e$", "$-z_e$")