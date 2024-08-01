import aerosandbox as asb
import aerosandbox.numpy as np
# from conventional import airplane
from lh2_airplane import airplane, mass_props_TOGW
from scipy import integrate

airplane = airplane.with_control_deflections(
    control_surface_deflection_mappings={
        "Elevator": -10,
    }
)

# integrator = "scipy"
integrator = "aerosandbox"

alpha_init = 1.3
V_init = 100

dyn_init = asb.DynamicsRigidBody3DBodyEuler(
    mass_props=mass_props_TOGW,
    x_e=0,
    y_e=0,
    z_e=0,
    u_b=V_init * np.cosd(alpha_init),
    v_b=0,
    w_b=V_init * np.sind(alpha_init),
    phi=0,
    theta=np.radians(alpha_init),
    psi=0,
    p=0,
    q=0,
    r=0
)

time = np.arange(0, 120, 0.5)

if integrator == "scipy":
    def dynamics(t, y):
        dyn = dyn_init.get_new_instance_with_state(dyn_init.pack_state(y))

        try:
            aero = asb.AeroBuildup(
                airplane=airplane,
                op_point=dyn.op_point,
                xyz_ref=dyn.mass_props.xyz_cg
            ).run()
        except FileNotFoundError:
            return np.nan * np.array(dyn.unpack_state(dyn.state))

        dyn.add_gravity_force()
        dyn.add_force(*aero["F_b"], axes="body")
        dyn.add_moment(*aero["M_b"], axes="body")

        derivatives = dyn.unpack_state(dyn.state_derivatives())

        print(t)
        print(dyn.op_point.__repr__())

        return derivatives


    sol = integrate.solve_ivp(
        fun=dynamics,
        t_span=(time.min(), time.max()),
        t_eval=time,
        y0=dyn_init.unpack_state(dyn_init.state),
        method="LSODA",
        rtol=1e100,
        atol=dyn_init.unpack_state(dict(
            x_e=1e-1,
            y_e=1e-1,
            z_e=1e-1,
            u_b=1e-2,
            v_b=1e-2,
            w_b=1e-2,
            phi=1e-2,
            theta=1e-2,
            psi=1e-2,
            p=1e-2,
            q=1e-2,
            r=1e-2,
        )),
        first_step=np.diff(time)[0],
        max_step=np.diff(time)[0],
        min_step=1e-2,
        # vectorized=True,
    )
    dyn = dyn_init.get_new_instance_with_state(dyn_init.pack_state(sol.y))

elif integrator == "aerosandbox":

    opti = asb.Opti()
    u_e, v_e, w_e = dyn_init.convert_axes(
        dyn_init.u_b, dyn_init.v_b, dyn_init.w_b,
        from_axes="body",
        to_axes="earth"
    )
    dyn = dyn_init.get_new_instance_with_state(
        dict(
            x_e=opti.variable(init_guess=time * u_e, scale=1e3),
            y_e=opti.variable(init_guess=time * v_e, scale=1e3),
            z_e=opti.variable(init_guess=time * w_e, scale=1e3),
            u_b=opti.variable(init_guess=dyn_init.u_b * np.ones_like(time), scale=1e2),
            v_b=opti.variable(init_guess=dyn_init.v_b * np.ones_like(time), scale=1e2),
            w_b=opti.variable(init_guess=dyn_init.w_b * np.ones_like(time), scale=1e2),
            phi=opti.variable(init_guess=dyn_init.phi * np.ones_like(time), scale=1e-1),
            theta=opti.variable(init_guess=dyn_init.theta * np.ones_like(time), scale=1e-1),
            psi=opti.variable(init_guess=dyn_init.psi * np.ones_like(time), scale=1e-1),
            p=opti.variable(init_guess=np.zeros_like(time), scale=1e-2),
            q=opti.variable(init_guess=np.zeros_like(time), scale=1e-2),
            r=opti.variable(init_guess=np.zeros_like(time), scale=1e-2),
        )
    )

    # dyn_for_aero = dyn.get_new_instance_with_state(dict(
    #     q=dyn.q * 10,
    # ))

    aero = asb.AeroBuildup(
        airplane=airplane,
        op_point=dyn.op_point,
        xyz_ref=dyn.mass_props.xyz_cg
    ).run_with_stability_derivatives()

    dyn.add_gravity_force()
    dyn.add_force(*aero["F_b"], axes="body")
    dyn.add_moment(*aero["M_b"], axes="body")

    dyn.constrain_derivatives(
        opti=opti,
        time=time,
    )

    opti.subject_to([
        v[0] == v_init
        for v, v_init in zip(dyn.state.values(), dyn_init.state.values())
    ])

    sol = opti.solve()

    dyn = sol(dyn)

else:
    raise ValueError

aero = asb.AeroBuildup(
    airplane=airplane,
    op_point=dyn.op_point,
    xyz_ref=dyn.mass_props.xyz_cg
).run_with_stability_derivatives()

import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

vars_to_plot = {
    **dyn.state,
    "alpha"   : dyn.alpha,
    "beta"    : dyn.beta,
    "speed"   : dyn.speed,
    "altitude": dyn.altitude,
    "CL"      : aero["CL"],
    "CD"      : aero["CD"],
    "Cm"      : aero["Cm"],
}
fig, axes = plt.subplots(6, 4, figsize=(15, 9), sharex=True)
for (k, v), ax in zip(vars_to_plot.items(), axes.flatten(order="F")):
    plt.sca(ax)
    plt.plot(time, v)
    plt.ylabel(k)
p.show_plot(dpi=80)

# fig, ax = plt.subplots()
# plt.plot(dyn.xe, dyn.altitude, "k")
# sc = plt.scatter(dyn.xe, dyn.altitude, c=dyn.speed, cmap=plt.cm.get_cmap("rainbow"), zorder=4)
# plt.axis('equal')
# plt.colorbar(label="Airspeed [m/s]")
# show_plot("Trajectory using `asb.AeroBuildup` Flight Dynamics", "$x_e$", "$-z_e$")
