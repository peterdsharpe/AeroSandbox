import aerosandbox as asb
import aerosandbox.numpy as np

if __name__ == '__main__':
    af = asb.Airfoil("naca0008", generate_polars=True)

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
    contour(Alpha, Re, CL, levels=30)
    plt.scatter(af.xfoil_data["alpha"], af.xfoil_data["Re"], color="k", alpha=0.2)
    plt.yscale('log')
    show_plot()

    ##### Plot Polars
    fig, ax = plt.subplots(2, 2, figsize=(8,8))
    Re=1e6
    alpha_lowres = np.linspace(-15, 15, 31)
    ma = alpha
    mCL = af.CL_function(alpha, Re)
    mCD = af.CD_function(alpha, Re)
    xf_run = asb.XFoil(af, Re=Re, max_iter=20).alpha(alpha_lowres)
    xa = xf_run["alpha"]
    xCL = xf_run["CL"]
    xCD = xf_run["CD"]

    plt.sca(ax[0,0])
    plt.plot(ma, mCL)
    plt.plot(xa, xCL, ".")

    plt.sca(ax[0,1])
    plt.plot(ma, mCD)
    plt.plot(xa, xCD, ".")
    plt.ylim(0, 0.05)

    plt.sca(ax[1, 0])
    plt.plot(mCD, mCL)
    plt.plot(xCD, xCL, ".")
    plt.xlim(0, 0.05)

    plt.sca(ax[1,1])
    plt.plot(ma, mCL/mCD)
    plt.plot(xa, xCL/xCD, ".")

    show_plot()

    ##### Test optimization
    opti = asb.Opti()
    alpha = opti.variable(init_guess=0, lower_bound=-20, upper_bound=20)
    opti.minimize(
        -af.CL_function(alpha, 1e6) / af.CD_function(alpha, 1e6)
    )
    sol = opti.solve()