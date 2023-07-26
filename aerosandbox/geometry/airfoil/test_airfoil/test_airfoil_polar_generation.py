import aerosandbox as asb
import aerosandbox.numpy as np

if __name__ == '__main__':
    af = asb.Airfoil("dae11")
    af.generate_polars()

    alpha = np.linspace(-40, 40, 300)
    re = np.geomspace(1e4, 1e12, 100)
    Alpha, Re = np.meshgrid(alpha, re)
    af.CL_function(alpha=0, Re=1e6)

    CL = af.CL_function(Alpha.flatten(), Re.flatten()).reshape(Alpha.shape)
    CD = af.CD_function(Alpha.flatten(), Re.flatten()).reshape(Alpha.shape)
    CM = af.CM_function(Alpha.flatten(), Re.flatten()).reshape(Alpha.shape)

    ##### Plot alpha-Re contours
    from aerosandbox.tools.pretty_plots import plt, show_plot, contour

    fig, ax = plt.subplots()
    contour(Alpha, Re, CL, levels=30, colorbar_label=r"$C_L$")
    plt.scatter(af.xfoil_data["alpha"], af.xfoil_data["Re"], color="k", alpha=0.2)
    plt.yscale('log')
    show_plot(
        f"Auto-generated Polar for {af.name} Airfoil",
        "Angle of Attack [deg]",
        "Reynolds Number [-]",
    )

    fig, ax = plt.subplots()
    contour(Alpha, Re, CD, levels=30, colorbar_label=r"$C_D$", z_log_scale=True)
    plt.scatter(af.xfoil_data["alpha"], af.xfoil_data["Re"], color="k", alpha=0.2)
    plt.yscale('log')
    show_plot(
        f"Auto-generated Polar for {af.name} Airfoil",
        "Angle of Attack [deg]",
        "Reynolds Number [-]",
    )

    ##### Plot Polars
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    Re = 1e6
    alpha_lowres = np.linspace(-15, 15, 31)
    ma = alpha
    mCL = af.CL_function(alpha, Re)
    mCD = af.CD_function(alpha, Re)
    xf_run = asb.XFoil(af, Re=Re, max_iter=20).alpha(alpha_lowres)
    xa = xf_run["alpha"]
    xCL = xf_run["CL"]
    xCD = xf_run["CD"]
    na = alpha
    nf_aero = af.get_aero_from_neuralfoil(
        alpha=na,
        Re=Re,
        mach=0,
    )


    plt.sca(ax[0, 0])
    plt.plot(ma, mCL, label="`Airfoil.generate_polars()`")
    plt.plot(na, nf_aero["CL"], label="NeuralFoil")
    plt.plot(xa, xCL, ".k")
    plt.xlabel("Angle of Attack [deg]")
    plt.ylabel("Lift Coefficient $C_L$ [-]")
    plt.legend(fontsize=8)

    plt.sca(ax[0, 1])
    plt.plot(ma, mCD)
    plt.plot(na, nf_aero["CD"])
    plt.plot(xa, xCD, ".k")
    plt.xlim(-10, 15)
    plt.ylim(0, 0.05)
    plt.xlabel("Angle of Attack [deg]")
    plt.ylabel("Drag Coefficient $C_D$ [-]")

    plt.sca(ax[1, 0])
    plt.plot(mCD, mCL)
    plt.plot(nf_aero["CD"], nf_aero["CL"])
    plt.plot(xCD, xCL, ".k")
    plt.xlim(0, 0.05)
    plt.xlabel("Drag Coefficient $C_D$ [-]")
    plt.ylabel("Lift Coefficient $C_L$ [-]")

    plt.sca(ax[1, 1])
    plt.plot(ma, mCL / mCD)
    plt.plot(na, nf_aero["CL"] / nf_aero["CD"])
    plt.plot(xa, xCL / xCD, ".k")
    plt.xlabel("Angle of Attack [deg]")
    plt.ylabel("Lift-to-Drag Ratio $C_L/C_D$ [-]")

    show_plot(legend=False)

    ##### Test optimization
    opti = asb.Opti()
    alpha = opti.variable(init_guess=0, lower_bound=-20, upper_bound=20)
    LD = af.CL_function(alpha, 1e6) / af.CD_function(alpha, 1e6)
    opti.minimize(
        -LD
    )
    sol = opti.solve()
